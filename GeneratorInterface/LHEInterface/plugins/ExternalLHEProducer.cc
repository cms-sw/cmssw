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
// $Id$
//
//


// system include files
#include <memory>
#include <string>
#include <unistd.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/wait.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class ExternalLHEProducer : public edm::EDProducer {
public:
  explicit ExternalLHEProducer(const edm::ParameterSet& iConfig);
  ~ExternalLHEProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  virtual void beginJob();
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  
  virtual void beginRun(edm::Run&, edm::EventSetup const&);
  virtual void endRun(edm::Run&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
  
  virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
  
  int closeDescriptors(int preserve);
  void executeScript();
  std::auto_ptr<std::string> readOutput();
  
  // ----------member data ---------------------------
  std::string scriptName_;
  std::string outputFile_;
  std::vector<std::string> args_;
  std::string outputContents_;
  
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
  args_(iConfig.getParameter<std::vector<std::string> >("args"))
{
  produces<std::string, edm::InLumi>("LHEScriptOutput"); 
}


ExternalLHEProducer::~ExternalLHEProducer()
{
  
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ExternalLHEProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}

// ------------ method called once each job just before starting event loop  ------------
void 
ExternalLHEProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ExternalLHEProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void 
ExternalLHEProducer::beginRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
ExternalLHEProducer::endRun(edm::Run&, edm::EventSetup const&)
{
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
std::auto_ptr<std::string> ExternalLHEProducer::readOutput()
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

  return std::auto_ptr<std::string>(new std::string(ss.str()));
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
ExternalLHEProducer::beginLuminosityBlock(edm::LuminosityBlock& lumi, edm::EventSetup const&)
{

  // pass luminosity block id as last argument, needed to define the seed of random number generators

  std::stringstream ss;
  ss << lumi.id().luminosityBlock();
  std::string iseed = ss.str();
  args_.push_back(iseed);

  executeScript();
  std::auto_ptr<std::string> localContents = readOutput();
  lumi.put(localContents, "LHEScriptOutput");
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
ExternalLHEProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
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

  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ExternalLHEProducer);
