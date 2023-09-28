// -*- C++ -*-
//
// Package:    ExternalLHEProducer
// Class:      ExternalLHEProducer
// 
/**\class ExternalLHEProducer ExternalLHEProducer.cc Example/ExternalLHEProducer/src/ExternalLHEProducer.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author:  Brian Paul Bockelman,8 R-018,+41227670861,
//         Created:  Fri Oct 21 11:37:26 CEST 2011
//
//


// system include files
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include "boost/filesystem.hpp"
#include <unistd.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "tbb/task_arena.h"


#include "boost/bind.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/ptr_container/ptr_deque.hpp"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Concurrency/interface/FunctorTask.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEXMLStringProduct.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHEReader.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class ExternalLHEProducer : public edm::one::EDProducer<edm::BeginRunProducer,
                                                        edm::EndRunProducer> {
public:
  explicit ExternalLHEProducer(const edm::ParameterSet& iConfig);
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:

  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRunProduce(edm::Run& run, edm::EventSetup const& es) override;
  void endRunProduce(edm::Run&, edm::EventSetup const&) override;
  void preallocThreads(unsigned int) override;

  std::vector<std::string> makeArgs(uint32_t nEvents, unsigned int nThreads, std::uint32_t seed) const;
  int closeDescriptors(int preserve) const;
  void executeScript(std::vector<std::string> const& args, int id, bool isPost) const;

  void nextEvent();
  
  // ----------member data ---------------------------
  std::string scriptName_;
  std::string outputFile_;
  const std::vector<std::string> args_;
  uint32_t npars_;
  uint32_t nEvents_;
  bool storeXML_;
  unsigned int nThreads_{1};
  std::string outputContents_;
  bool generateConcurrently_{false};
  const std::vector<std::string> postGenerationCommand_;

  // Used only if nPartonMapping is in the configuration
  std::map<unsigned, std::pair<unsigned, unsigned>> nPartonMapping_{};

  std::unique_ptr<lhef::LHEReader>	reader_;
  std::shared_ptr<lhef::LHERunInfo> runInfoLast_;
  std::shared_ptr<lhef::LHERunInfo> runInfo_;
  std::shared_ptr<lhef::LHEEvent> partonLevel_;
  boost::ptr_deque<LHERunInfoProduct> runInfoProducts_;
  bool					wasMerged;
  
  class FileCloseSentry : private boost::noncopyable {
  public:
    explicit FileCloseSentry(int fd) : fd_(fd) {};
    
    ~FileCloseSentry() {
      close(fd_);
    }
  private:
    int fd_;
  };
 
};

//
// constructors and destructor
//
ExternalLHEProducer::ExternalLHEProducer(const edm::ParameterSet& iConfig) :
  scriptName_((iConfig.getParameter<edm::FileInPath>("scriptName")).fullPath()),
  outputFile_(iConfig.getParameter<std::string>("outputFile")),
  args_(iConfig.getParameter<std::vector<std::string> >("args")),
  npars_(iConfig.getParameter<uint32_t>("numberOfParameters")),
  nEvents_(iConfig.getUntrackedParameter<uint32_t>("nEvents")),
  storeXML_(iConfig.getUntrackedParameter<bool>("storeXML")),
  generateConcurrently_(iConfig.getUntrackedParameter<bool>("generateConcurrently")),
  postGenerationCommand_(iConfig.getUntrackedParameter<std::vector<std::string>>("postGenerationCommand"))
{
  if (npars_ != args_.size())
    throw cms::Exception("ExternalLHEProducer") << "Problem with configuration: " << args_.size() << " script arguments given, expected " << npars_;

  if (iConfig.exists("nPartonMapping")) {
    auto& processMap(iConfig.getParameterSetVector("nPartonMapping"));
    for (auto& cfg : processMap) {
      unsigned processId(cfg.getParameter<unsigned>("idprup"));

      auto orderStr(cfg.getParameter<std::string>("order"));
      unsigned order(0);
      if (orderStr == "LO")
        order = 0;
      else if (orderStr == "NLO")
        order = 1;
      else
        throw cms::Exception("ExternalLHEProducer") << "Invalid order specification for process " << processId << ": " << orderStr;
      
      unsigned np(cfg.getParameter<unsigned>("np"));
      
      nPartonMapping_.emplace(processId, std::make_pair(order, np));
    }
  }

  produces<LHEXMLStringProduct, edm::Transition::BeginRun>("LHEScriptOutput"); 

  produces<LHEEventProduct>();
  produces<LHERunInfoProduct, edm::Transition::BeginRun>();
  produces<LHERunInfoProduct, edm::Transition::EndRun>();
}


//
// member functions
//

// ------------ method called with number of threads in job --
void
ExternalLHEProducer::preallocThreads(unsigned int iThreads)
{
  nThreads_ = iThreads;
}

// ------------ method called to produce the data  ------------
void
ExternalLHEProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  nextEvent();
  if (!partonLevel_) {
    throw edm::Exception(edm::errors::EventGenerationFailure) << "No lhe event found in ExternalLHEProducer::produce().  "
    << "The likely cause is that the lhe file contains fewer events than were requested, which is possible "
    << "in case of phase space integration or unweighting efficiency problems.";
  }

  std::unique_ptr<LHEEventProduct> product(
	       new LHEEventProduct(*partonLevel_->getHEPEUP(),
				   partonLevel_->originalXWGTUP())
	       );
  if (partonLevel_->getPDF()) {
    product->setPDF(*partonLevel_->getPDF());
  }
  std::for_each(partonLevel_->weights().begin(),
                partonLevel_->weights().end(),
                boost::bind(&LHEEventProduct::addWeight,
                            product.get(), _1));
  product->setScales(partonLevel_->scales());
  product->setEvtNum(partonLevel_->evtnum());
  if (nPartonMapping_.empty()) {
    product->setNpLO(partonLevel_->npLO());
    product->setNpNLO(partonLevel_->npNLO());
  }
  else {
    // overwrite npLO and npNLO values by user-specified mapping
    unsigned processId(partonLevel_->getHEPEUP()->IDPRUP);
    unsigned order(0);
    unsigned np(0);
    try {
      auto procDef(nPartonMapping_.at(processId));
      order = procDef.first;
      np = procDef.second;
    }
    catch (std::out_of_range&) {
      throw cms::Exception("ExternalLHEProducer") << "Unexpected IDPRUP encountered: " << partonLevel_->getHEPEUP()->IDPRUP;
    }

    switch (order) {
    case 0:
      product->setNpLO(np);
      product->setNpNLO(-1);
      break;
    case 1:
      product->setNpLO(-1);
      product->setNpNLO(np);
      break;
    default:
      break;
    }
  }

  std::for_each(partonLevel_->getComments().begin(),
                partonLevel_->getComments().end(),
                boost::bind(&LHEEventProduct::addComment,
                            product.get(), _1));

  iEvent.put(std::move(product));

  if (runInfo_) {
    std::unique_ptr<LHERunInfoProduct> product(new LHERunInfoProduct(*runInfo_->getHEPRUP()));
    std::for_each(runInfo_->getHeaders().begin(),
                  runInfo_->getHeaders().end(),
                  boost::bind(&LHERunInfoProduct::addHeader,
                              product.get(), _1));
    std::for_each(runInfo_->getComments().begin(),
                  runInfo_->getComments().end(),
                  boost::bind(&LHERunInfoProduct::addComment,
                              product.get(), _1));
  
    if (!runInfoProducts_.empty()) {
      runInfoProducts_.front().mergeProduct(*product);
      if (!wasMerged) {
        runInfoProducts_.pop_front();
        runInfoProducts_.push_front(product.release());
        wasMerged = true;
      }
    }
  
    runInfo_.reset();
  }
  
  partonLevel_.reset();
  return; 
}

// ------------ method called when starting to processes a run  ------------
void 
ExternalLHEProducer::beginRunProduce(edm::Run& run, edm::EventSetup const& es)
{

  // pass the number of events as previous to last argument
  
  // pass the random number generator seed as last argument

  edm::Service<edm::RandomNumberGenerator> rng;

  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "The ExternalLHEProducer module requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file if you want to run ExternalLHEProducer";
  }

  std::vector<std::string> infiles;
  auto const seed = rng->mySeed();
  if (generateConcurrently_) {
    infiles.resize(nThreads_);
    auto const nEventsAve = nEvents_ / nThreads_;
    unsigned int const overflow = nThreads_ - (nEvents_ % nThreads_);
    std::exception_ptr except;
    std::atomic<char> exceptSet{0};

    tbb::this_task_arena::isolate([this, &except, &infiles, &exceptSet, nEventsAve, overflow, seed]() {
      tbb::empty_task* waitTask = new (tbb::task::allocate_root()) tbb::empty_task;
      waitTask->set_ref_count(1 + nThreads_);

      for (unsigned int t = 0; t < nThreads_; ++t) {
        uint32_t nEvents = nEventsAve;
        if (nEvents_ % nThreads_ != 0 and t >= overflow) {
          nEvents += 1;
        }
        auto task = edm::make_functor_task(tbb::task::allocate_root(),
                                           [t, this, &infiles, seed, nEvents, &except, &exceptSet, waitTask]() {
                                             try {
                                               using namespace boost::filesystem;
                                               using namespace std::string_literals;
                                               auto out = path("thread"s + std::to_string(t)) / path(outputFile_);
                                               infiles[t] = out.native();
                                               executeScript(makeArgs(nEvents, 1, seed + t), t, false);
                                             } catch (...) {
                                               char expected = 0;
                                               if (exceptSet.compare_exchange_strong(expected, 1)) {
                                                 except = std::current_exception();
                                                 exceptSet.store(2);
                                               }
                                             }
                                             waitTask->decrement_ref_count();
                                           });
        tbb::task::spawn(*task);
      }
      waitTask->wait_for_all();
      tbb::task::destroy(*waitTask);
    });
    if (exceptSet != 0) {
      std::rethrow_exception(except);
    }
  } else {
    infiles = std::vector<std::string>(1, outputFile_);
    executeScript(makeArgs(nEvents_, nThreads_, seed), 0, false);
  }

  //run post-generation command if specified
  if (!postGenerationCommand_.empty()) {
    std::vector<std::string> postcmd = postGenerationCommand_;
    try {
      postcmd[0] = edm::FileInPath(postcmd[0]).fullPath();
    } catch (const edm::Exception& e) {
      edm::LogWarning("ExternalLHEProducer") << postcmd[0] << " is not a relative path. Run it as a shell command.";
    }
    executeScript(postcmd, 0, true);
  }

  //fill LHEXMLProduct (streaming read directly into compressed buffer to save memory)
  std::unique_ptr<LHEXMLStringProduct> p(new LHEXMLStringProduct);

  //store the XML file only if explictly requested
  if (storeXML_) {
    std::string file;
    if (generateConcurrently_) {
      using namespace boost::filesystem;
      file = (path("thread0") / path(outputFile_)).native();
    } else {
      file = outputFile_;
    }
    std::ifstream instream(file);
    if (!instream) {
      throw cms::Exception("OutputOpenError") << "Unable to open script output file " << outputFile_ << ".";
    }  
    instream.seekg (0, instream.end);
    int insize = instream.tellg();
    instream.seekg (0, instream.beg);  
    p->fillCompressedContent(instream, 0.25*insize);
    instream.close();
  }
  run.put(std::move(p), "LHEScriptOutput");

  // LHE C++ classes translation
  // (read back uncompressed file from disk in streaming mode again to save memory)

  unsigned int skip = 0;
  reader_ = std::make_unique<lhef::LHEReader>(infiles, skip);

  nextEvent();
  if (runInfoLast_) {
    runInfo_ = runInfoLast_;
  
    std::unique_ptr<LHERunInfoProduct> product(new LHERunInfoProduct(*runInfo_->getHEPRUP()));
    std::for_each(runInfo_->getHeaders().begin(),
                  runInfo_->getHeaders().end(),
                  boost::bind(&LHERunInfoProduct::addHeader,
                              product.get(), _1));
    std::for_each(runInfo_->getComments().begin(),
                  runInfo_->getComments().end(),
                  boost::bind(&LHERunInfoProduct::addComment,
                              product.get(), _1));
  
    // keep a copy around in case of merging
    runInfoProducts_.push_back(new LHERunInfoProduct(*product));
    wasMerged = false;
  
    run.put(std::move(product));
  
    runInfo_.reset();
  }

}

// ------------ method called when ending the processing of a run  ------------
void 
ExternalLHEProducer::endRunProduce(edm::Run& run, edm::EventSetup const& es)
{

  if (!runInfoProducts_.empty()) {
    std::unique_ptr<LHERunInfoProduct> product(runInfoProducts_.pop_front().release());
    run.put(std::move(product));
  }
  
  nextEvent();
  if (partonLevel_) {
    // VALIDATION_RUN env variable allows to finish event processing early without errors by sending SIGINT
    if (std::getenv("VALIDATION_RUN") != nullptr) {
      edm::LogWarning("ExternalLHEProducer")
          << "Event loop is over, but there are still lhe events to process, ignoring...";
    } else {
      throw edm::Exception(edm::errors::EventGenerationFailure)
          << "Error in ExternalLHEProducer::endRunProduce().  "
          << "Event loop is over, but there are still lhe events to process."
          << "This could happen if lhe file contains more events than requested.  This is never expected to happen.";
    }
  }  
  
  reader_.reset();  
  if (generateConcurrently_) {
    for (unsigned int t = 0; t < nThreads_; ++t) {
      using namespace boost::filesystem;
      using namespace std::string_literals;
      auto out = path("thread"s + std::to_string(t)) / path(outputFile_);
      if (unlink(out.c_str())) {
        throw cms::Exception("OutputDeleteError") << "Unable to delete original script output file " << out
                                                  << " (errno=" << errno << ", " << strerror(errno) << ").";
      }
    }
  } else {
    if (unlink(outputFile_.c_str())) {
      throw cms::Exception("OutputDeleteError") << "Unable to delete original script output file " << outputFile_
                                                << " (errno=" << errno << ", " << strerror(errno) << ").";
    }
  }
}

std::vector<std::string> ExternalLHEProducer::makeArgs(uint32_t nEvents,
                                                       unsigned int nThreads,
                                                       std::uint32_t seed) const {
  std::vector<std::string> args;
  args.reserve(3 + args_.size());

  args.push_back(args_.front());
  args.push_back(std::to_string(nEvents));

  args.push_back(std::to_string(seed));

  args.push_back(std::to_string(nThreads));
  std::copy(args_.begin() + 1, args_.end(), std::back_inserter(args));

  for (unsigned int iArg = 0; iArg < args.size(); iArg++) {
    LogDebug("LHEInputArgs") << "arg [" << iArg << "] = " << args[iArg];
  }

  return args;
}

// ------------ Close all the open file descriptors ------------
int
ExternalLHEProducer::closeDescriptors(int preserve) const
{
  int maxfd = 1024;
  int fd;
#ifdef __linux__
  DIR * dir;
  struct dirent *dp;
  maxfd = preserve;
  if ((dir = opendir("/proc/self/fd"))) {
    errno = 0;
    while ((dp = readdir (dir)) != nullptr) {
      if ((strcmp(dp->d_name, ".") == 0)  || (strcmp(dp->d_name, "..") == 0)) {
        continue;
      }
      if (sscanf(dp->d_name, "%d", &fd) != 1) {
        //throw cms::Exception("closeDescriptors") << "Found unexpected filename in /proc/self/fd: " << dp->d_name;
        return -1;
      }
      if (fd > maxfd) {
        maxfd = fd;
      }
    }
    if (errno) {
      //throw cms::Exception("closeDescriptors") << "Unable to determine the number of fd (errno=" << errno << ", " << strerror(errno) << ").";
      return errno;
    }
    closedir(dir);
  }
#endif
  // TODO: assert for an unreasonable number of fds?
  for (fd=3; fd<maxfd+1; fd++) {
    if (fd != preserve)
      close(fd);
  }
  return 0;
}

// ------------ Execute the script associated with this producer ------------
void 
ExternalLHEProducer::executeScript(std::vector<std::string> const& args, int id, bool isPost) const
{

  // Fork a script, wait until it finishes.

  int rc = 0, rc2 = 0;
  int filedes[2], fd_flags;

  if (pipe(filedes)) {
    throw cms::Exception("Unable to create a new pipe");
  }
  FileCloseSentry sentry1(filedes[0]), sentry2(filedes[1]);

  if ((fd_flags = fcntl(filedes[1], F_GETFD, NULL)) == -1) {
    throw cms::Exception("ExternalLHEProducer") << "Failed to get pipe file descriptor flags (errno=" << rc << ", " << strerror(rc) << ")";
  }
  if (fcntl(filedes[1], F_SETFD, fd_flags | FD_CLOEXEC) == -1) {
    throw cms::Exception("ExternalLHEProducer") << "Failed to set pipe file descriptor flags (errno=" << rc << ", " << strerror(rc) << ")";
  }

  unsigned int argc_pre = 0;
  // For generation command the first argument gives to the scriptName
  if (!isPost) {
    argc_pre = 1;
  }
  unsigned int argc = argc_pre + args.size();
  // TODO: assert that we have a reasonable number of arguments
  char **argv = new char *[argc+1];
  if (!isPost) {
    argv[0] = strdup(scriptName_.c_str());
  }
  for (unsigned int i = 0; i < args.size(); i++) {
    argv[argc_pre + i] = strdup(args[i].c_str());
  }
  argv[argc] = nullptr;

  pid_t pid = fork();
  if (pid == 0) {
    // The child process
    if (!(rc = closeDescriptors(filedes[1]))) {
      if (!isPost && generateConcurrently_) {
        using namespace boost::filesystem;
        using namespace std::string_literals;
        boost::system::error_code ec;
        auto newDir = path("thread"s + std::to_string(id));
        create_directory(newDir, ec);
        current_path(newDir, ec);
      }
      execvp(argv[0], argv); // If execv returns, we have an error.
      rc = errno;
    }
    while ((write(filedes[1], &rc, sizeof(int)) == -1) && (errno == EINTR)) {}
    _exit(1);
  }

  // Free the arg vector ASAP
  for (unsigned int i=0; i<args.size()+1; i++) {
    free(argv[i]);
  }
  delete [] argv;

  if (pid == -1) {
    throw cms::Exception("ForkException") << "Unable to fork a child (errno=" << errno << ", " << strerror(errno) << ")";
  }

  close(filedes[1]);
  // If the exec succeeds, the read will fail.
  while (((rc2 = read(filedes[0], &rc, sizeof(int))) == -1) && (errno == EINTR)) { rc2 = 0; }
  if ((rc2 == sizeof(int)) && rc) {
    throw cms::Exception("ExternalLHEProducer") << "Failed to execute script (errno=" << rc << ", " << strerror(rc) << ")";
  }
  close(filedes[0]);

  int status = 0;
  errno = 0;
  do {
    if (waitpid(pid, &status, 0) < 0) {
      if (errno == EINTR) {
        continue;
      } else {
        throw cms::Exception("ExternalLHEProducer") << "Failed to read child status (errno=" << errno << ", " << strerror(errno) << ")";
      }
    }
    if (WIFSIGNALED(status)) {
      throw cms::Exception("ExternalLHEProducer") << "Child exited due to signal " << WTERMSIG(status) << ".";
    }
    if (WIFEXITED(status)) {
      rc = WEXITSTATUS(status);
      break;
    }
  } while (true);
  if (rc) {
    throw cms::Exception("ExternalLHEProducer") << "Child failed with exit code " << rc << ".";
  }

}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
ExternalLHEProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setComment("Executes an external script and places its output file into an EDM collection");

  edm::FileInPath thePath;
  desc.add<edm::FileInPath>("scriptName", thePath);
  desc.add<std::string>("outputFile", "myoutput");
  desc.add<std::vector<std::string> >("args");
  desc.add<uint32_t>("numberOfParameters");
  desc.addUntracked<uint32_t>("nEvents");
  desc.addUntracked<bool>("storeXML", false);
  desc.addUntracked<bool>("generateConcurrently", false)
      ->setComment("If true, run the script concurrently in separate processes.");
  desc.addUntracked<std::vector<std::string>>("postGenerationCommand", std::vector<std::string>())
      ->setComment(
          "Command to run after the generation script has completed. The first argument can be a relative path.");

  edm::ParameterSetDescription nPartonMappingDesc;
  nPartonMappingDesc.add<unsigned>("idprup");
  nPartonMappingDesc.add<std::string>("order");
  nPartonMappingDesc.add<unsigned>("np");
  desc.addVPSetOptional("nPartonMapping", nPartonMappingDesc);

  descriptions.addDefault(desc);
}

void ExternalLHEProducer::nextEvent()
{

  if (partonLevel_)
    return;

  if(not reader_) { return;}

  partonLevel_ = reader_->next();
  if (!partonLevel_) {
    //see if we have another file to read;
    bool newFileOpened;
    do {
      newFileOpened = false;
      partonLevel_ = reader_->next(&newFileOpened);
    } while (newFileOpened && !partonLevel_);
  }
  if (!partonLevel_)
    return;

  std::shared_ptr<lhef::LHERunInfo> runInfoThis = partonLevel_->getRunInfo();
  if (runInfoThis != runInfoLast_) {
    runInfo_ = runInfoThis;
    runInfoLast_ = runInfoThis;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(ExternalLHEProducer);
