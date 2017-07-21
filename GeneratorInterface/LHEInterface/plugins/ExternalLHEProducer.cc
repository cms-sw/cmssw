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
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <unistd.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/wait.h>

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
  virtual ~ExternalLHEProducer() override;
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:

  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void beginRunProduce(edm::Run& run, edm::EventSetup const& es) override;
  virtual void endRunProduce(edm::Run&, edm::EventSetup const&) override;
  virtual void preallocThreads(unsigned int) override;

  int closeDescriptors(int preserve);
  void executeScript();
  std::unique_ptr<std::string> readOutput();

  void nextEvent();
  
  // ----------member data ---------------------------
  std::string scriptName_;
  std::string outputFile_;
  std::vector<std::string> args_;
  uint32_t npars_;
  uint32_t nEvents_;
  unsigned int nThreads_{1};
  std::string outputContents_;

  std::unique_ptr<lhef::LHEReader>	reader_;
  boost::shared_ptr<lhef::LHERunInfo>	runInfoLast;
  boost::shared_ptr<lhef::LHERunInfo>	runInfo;
  boost::shared_ptr<lhef::LHEEvent>	partonLevel;
  boost::ptr_deque<LHERunInfoProduct>	runInfoProducts;
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
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
ExternalLHEProducer::ExternalLHEProducer(const edm::ParameterSet& iConfig) :
  scriptName_((iConfig.getParameter<edm::FileInPath>("scriptName")).fullPath().c_str()),
  outputFile_(iConfig.getParameter<std::string>("outputFile")),
  args_(iConfig.getParameter<std::vector<std::string> >("args")),
  npars_(iConfig.getParameter<uint32_t>("numberOfParameters")),
  nEvents_(iConfig.getUntrackedParameter<uint32_t>("nEvents"))
{
  if (npars_ != args_.size())
    throw cms::Exception("ExternalLHEProducer") << "Problem with configuration: " << args_.size() << " script arguments given, expected " << npars_;
  produces<LHEXMLStringProduct, edm::Transition::BeginRun>("LHEScriptOutput"); 

  produces<LHEEventProduct>();
  produces<LHERunInfoProduct, edm::Transition::BeginRun>();
  //produces<LHERunInfoProduct, edm::Transition::EndRun>();
}


ExternalLHEProducer::~ExternalLHEProducer()
{
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
  if (!partonLevel) {
    throw cms::Exception("ExternalLHEProducer") << "No lhe event found in ExternalLHEProducer::produce().  "
    << "The likely cause is that the lhe file contains fewer events than were requested, which is possible "
    << "in case of phase space integration or uneweighting efficiency problems.";
  }

  std::unique_ptr<LHEEventProduct> product(
	       new LHEEventProduct(*partonLevel->getHEPEUP(),
				   partonLevel->originalXWGTUP())
	       );
  if (partonLevel->getPDF()) {
    product->setPDF(*partonLevel->getPDF());
  }
  std::for_each(partonLevel->weights().begin(),
                partonLevel->weights().end(),
                boost::bind(&LHEEventProduct::addWeight,
                            product.get(), _1));
  product->setScales(partonLevel->scales());
  product->setNpLO(partonLevel->npLO());
  product->setNpNLO(partonLevel->npNLO());
  std::for_each(partonLevel->getComments().begin(),
                partonLevel->getComments().end(),
                boost::bind(&LHEEventProduct::addComment,
                            product.get(), _1));

  iEvent.put(std::move(product));

  if (runInfo) {
    std::auto_ptr<LHERunInfoProduct> product(new LHERunInfoProduct(*runInfo->getHEPRUP()));
    std::for_each(runInfo->getHeaders().begin(),
                  runInfo->getHeaders().end(),
                  boost::bind(&LHERunInfoProduct::addHeader,
                              product.get(), _1));
    std::for_each(runInfo->getComments().begin(),
                  runInfo->getComments().end(),
                  boost::bind(&LHERunInfoProduct::addComment,
                              product.get(), _1));
  
    if (!runInfoProducts.empty()) {
      runInfoProducts.front().mergeProduct(*product);
      if (!wasMerged) {
        runInfoProducts.pop_front();
        runInfoProducts.push_front(product);
        wasMerged = true;
      }
    }
  
    runInfo.reset();
  }
  
  partonLevel.reset();
  return; 
}

// ------------ method called when starting to processes a run  ------------
void 
ExternalLHEProducer::beginRunProduce(edm::Run& run, edm::EventSetup const& es)
{

  // pass the number of events as previous to last argument
  
  std::ostringstream eventStream;
  eventStream << nEvents_;
  // args_.push_back(eventStream.str());
  args_.insert(args_.begin() + 1, eventStream.str());

  // pass the random number generator seed as last argument

  edm::Service<edm::RandomNumberGenerator> rng;

  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "The ExternalLHEProducer module requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file if you want to run ExternalLHEProducer";
  }
  std::ostringstream randomStream;
  randomStream << rng->mySeed(); 
  // args_.push_back(randomStream.str());
  args_.insert(args_.begin() + 2, randomStream.str());

  // args_.emplace_back(std::to_string(nThreads_));
  args_.insert(args_.begin() + 3, std::to_string(nThreads_));

  for ( unsigned int iArg = 0; iArg < args_.size() ; iArg++ ) {
    LogDebug("LHEInputArgs") << "arg [" << iArg << "] = " << args_[iArg];
  }

  executeScript();
  
  //fill LHEXMLProduct (streaming read directly into compressed buffer to save memory)
  std::unique_ptr<LHEXMLStringProduct> p(new LHEXMLStringProduct);
  std::ifstream instream(outputFile_);
  if (!instream) {
    throw cms::Exception("OutputOpenError") << "Unable to open script output file " << outputFile_ << ".";
  }  
  instream.seekg (0, instream.end);
  int insize = instream.tellg();
  instream.seekg (0, instream.beg);  
  p->fillCompressedContent(instream, 0.25*insize);
  instream.close();
  run.put(std::move(p), "LHEScriptOutput");

  // LHE C++ classes translation
  // (read back uncompressed file from disk in streaming mode again to save memory)

  std::vector<std::string> infiles(1, outputFile_);
  unsigned int skip = 0;
  reader_ = std::make_unique<lhef::LHEReader>(infiles, skip);

  nextEvent();
  if (runInfoLast) {
    runInfo = runInfoLast;
  
    std::unique_ptr<LHERunInfoProduct> product(new LHERunInfoProduct(*runInfo->getHEPRUP()));
    std::for_each(runInfo->getHeaders().begin(),
                  runInfo->getHeaders().end(),
                  boost::bind(&LHERunInfoProduct::addHeader,
                              product.get(), _1));
    std::for_each(runInfo->getComments().begin(),
                  runInfo->getComments().end(),
                  boost::bind(&LHERunInfoProduct::addComment,
                              product.get(), _1));
  
    // keep a copy around in case of merging
    runInfoProducts.push_back(new LHERunInfoProduct(*product));
    wasMerged = false;
  
    run.put(std::move(product));
  
    runInfo.reset();
  }

}

// ------------ method called when ending the processing of a run  ------------
void 
ExternalLHEProducer::endRunProduce(edm::Run& run, edm::EventSetup const& es)
{

  if (!runInfoProducts.empty()) {
    std::unique_ptr<LHERunInfoProduct> product(runInfoProducts.pop_front().release());
    run.put(std::move(product));
  }
  
  nextEvent();
  if (partonLevel) {
    throw cms::Exception("ExternalLHEProducer") << "Error in ExternalLHEProducer::endRunProduce().  "
    << "Event loop is over, but there are still lhe events to process."
    << "This could happen if lhe file contains more events than requested.  This is never expected to happen.";
  }  
  
  reader_.reset();  
  
  if (unlink(outputFile_.c_str())) {
    throw cms::Exception("OutputDeleteError") << "Unable to delete original script output file " << outputFile_ << " (errno=" << errno << ", " << strerror(errno) << ").";
  }  

}

// ------------ Close all the open file descriptors ------------
int
ExternalLHEProducer::closeDescriptors(int preserve)
{
  int maxfd = 1024;
  int fd;
#ifdef __linux__
  DIR * dir;
  struct dirent *dp;
  maxfd = preserve;
  if ((dir = opendir("/proc/self/fd"))) {
    errno = 0;
    while ((dp = readdir (dir)) != NULL) {
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
ExternalLHEProducer::executeScript()
{

  // Fork a script, wait until it finishes.

  int rc = 0, rc2 = 0;
  int filedes[2], fd_flags;
  unsigned int argc;

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

  argc = 1 + args_.size();
  // TODO: assert that we have a reasonable number of arguments
  char **argv = new char *[argc+1];
  argv[0] = strdup(scriptName_.c_str());
  for (unsigned int i=1; i<argc; i++) {
    argv[i] = strdup(args_[i-1].c_str());
  }
  argv[argc] = NULL;

  pid_t pid = fork();
  if (pid == 0) {
    // The child process
    if (!(rc = closeDescriptors(filedes[1]))) {
      execvp(argv[0], argv); // If execv returns, we have an error.
      rc = errno;
    }
    while ((write(filedes[1], &rc, sizeof(int)) == -1) && (errno == EINTR)) {}
    _exit(1);
  }

  // Free the arg vector ASAP
  for (unsigned int i=0; i<args_.size()+1; i++) {
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

// ------------ Read the output script ------------
#define BUFSIZE 4096
std::unique_ptr<std::string> ExternalLHEProducer::readOutput()
{
  int fd;
  ssize_t n;
  char buf[BUFSIZE];

  if ((fd = open(outputFile_.c_str(), O_RDONLY)) == -1) {
    throw cms::Exception("OutputOpenError") << "Unable to open script output file " << outputFile_ << " (errno=" << errno << ", " << strerror(errno) << ").";
  }

  std::stringstream ss;
  while ((n = read(fd, buf, BUFSIZE)) > 0 || (n == -1 && errno == EINTR)) {
    if (n > 0)
      ss.write(buf, n);
  }
  if (n == -1) {
    throw cms::Exception("OutputOpenError") << "Unable to read from script output file " << outputFile_ << " (errno=" << errno << ", " << strerror(errno) << ").";
  }

  if (unlink(outputFile_.c_str())) {
    throw cms::Exception("OutputDeleteError") << "Unable to delete original script output file " << outputFile_ << " (errno=" << errno << ", " << strerror(errno) << ").";
  }

  return std::unique_ptr<std::string>(new std::string(ss.str()));
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

  descriptions.addDefault(desc);
}

void ExternalLHEProducer::nextEvent()
{

  if (partonLevel)
    return;

  if(not reader_) { return;}
  partonLevel = reader_->next();
  if (!partonLevel)
    return;

  boost::shared_ptr<lhef::LHERunInfo> runInfoThis = partonLevel->getRunInfo();
  if (runInfoThis != runInfoLast) {
    runInfo = runInfoThis;
    runInfoLast = runInfoThis;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(ExternalLHEProducer);
