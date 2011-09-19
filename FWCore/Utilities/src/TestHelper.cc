//------------------------------------------------------------
//------------------------------------------------------------
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include <sys/wait.h>
#include <unistd.h>
#include <cstring>

#include "boost/version.hpp"
#include "boost/filesystem/convenience.hpp"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TestHelper.h"

namespace bf = boost::filesystem;

int run_script(std::string const& shell, std::string const& script) {
  pid_t pid = 0;
  int status = 0;

  if((pid = fork()) < 0) {
      std::cerr << "fork failed, to run " << script << std::endl;;
      return -1;
  }

  if(pid == 0) { // child
    execlp(shell.c_str(), "sh", "-c", script.c_str(), static_cast<char const*>(0));
    std::cerr << "child failed becuase '" << strerror(errno) << "'\n";
    _exit(127); // signal parent and children processes
  } else { // parent
    while(waitpid(pid, &status, 0) < 0) {
      if(errno != EINTR) {
        std::cerr << "child process failed " << strerror(errno) << "\n";
        status = -1;
        break;
      } else {
        if(WIFSIGNALED(status)) {
          std::cerr << "child existed because of a signal " << WTERMSIG(status) << "\n";
        }
      }
    }
    if(WIFSIGNALED(status)) {
      std::cerr << "child existed because of a signal " << WTERMSIG(status) << "\n";
    }
    if(WIFEXITED(status)) {
    }
  }
  return status;
}

int do_work(int argc, char* argv[], char** env) {
  bf::path currentPath(bf::initial_path().string());

  if(argc < 4) {
    std::cout << "Usage: " << argv[0] << " shell subdir script1 script2 ... scriptN\n\n"
              << "where shell is the path+shell (e.g., /bin/bash) intended to run the scripts\n"
              << "and subdir is the subsystem/package/subdir in which the scripts are found\n"
              << "(e.g., FWCore/Utilities/test)\n"
              << std::endl;

    std::cout << "Current directory is: " << currentPath.string() << '\n';
    std::cout << "Current environment:\n";
    std::cout << "---------------------\n";
    for(int i = 0; env[i] != 0; ++i) std::cout << env[i] << '\n';
    std::cout << "---------------------\n";
    std::cout << "Executable name: " << argv[0] << '\n';
    return -1;
  }

  for(int i = 0; i < argc; ++i) {
    std::cout << "argument " << i << ": " << argv[i] << '\n';
  }

  std::string shell(argv[1]);
  std::cerr << "shell is: " << shell << '\n';

  std::cout << "Current directory is: " << currentPath.string() << '\n';
  // It is unclear about which of these environment variables should
  // be used.
  char const* topdir = getenv("SCRAMRT_LOCALRT");
  if(!topdir) topdir = getenv("LOCALRT");

  char const* arch = getenv("SCRAM_ARCH");

  if(!arch) {
    // Try to synthesize SCRAM_ARCH value.
    bf::path exepath(argv[0]);
#if (BOOST_VERSION / 100000) >= 1 && ((BOOST_VERSION / 100) % 1000) >= 47
    std::string maybe_arch = exepath.parent_path().filename().string();
#else
    std::string maybe_arch = exepath.branch_path().leaf();
#endif
    if(setenv("SCRAM_ARCH", maybe_arch.c_str(), 1) != 0) {
      std::cerr << "SCRAM_ARCH not set and attempt to set it failed\n";
      return -1;
    }
    arch = getenv("SCRAM_ARCH");
  }

  int rc = 0;

  if(!topdir) {
    std::cout << "Neither SCRAMRT_LOCALRT nor LOCALRT is not defined" << std::endl;;
    return -1;
  }


  std::string testdir(topdir); testdir += "/src/"; testdir += argv[2];
  std::string tmpdir(topdir);  tmpdir += "/tmp/";   tmpdir += arch;
  std::string testbin(topdir); testbin += "/test/"; testbin += arch;

  std::cout << "topdir is: " << topdir << '\n';
  std::cout << "testdir is: " << testdir << '\n';
  std::cout << "tmpdir is: " << tmpdir << '\n';
  std::cout << "testbin is: " << testbin << '\n';

  if(setenv("LOCAL_TEST_DIR", testdir.c_str(),1) != 0) {
    std::cerr << "Could not set LOCAL_TEST_DIR to " << testdir << std::endl;;
    return -1;
  }
  if(setenv("LOCAL_TMP_DIR", tmpdir.c_str(),1) != 0) {
    std::cerr << "Could not set LOCAL_TMP_DIR to " << tmpdir << std::endl;;
    return -1;
  }
  if(setenv("LOCAL_TOP_DIR", topdir,1) != 0) {
    std::cerr << "Could not set LOCAL_TOP_DIR to " << topdir << std::endl;;
    return -1;
  }
  if(setenv("LOCAL_TEST_BIN", testbin.c_str(),1) != 0) {
    std::cerr << "Could not set LOCAL_TEST_BIN to " << testbin << std::endl;;
    return -1;
  }

  testdir += "/";

  for(int i = 3; i < argc && rc == 0; ++i) {
    std::string scriptname(testdir);
    scriptname += argv[i];
    std::cout << "Running script: " << scriptname << std::endl;
    rc = run_script(shell, scriptname);
  }

  std::cout << "status = " << rc << std::endl;
  return rc == 0 ? 0 : -1;
}

int ptomaine(int argc, char* argv[], char** env) {
  int rc = 1;
  try {
    rc = do_work(argc, argv, env);
  }
  catch(edm::Exception& x) {
    std::cerr << "Caught an edm::Exception in "
              << argv[0] << '\n'
              << x;
  }
  catch(cms::Exception& x) {
    std::cerr << "Caught a cms::Exception in "
              << argv[0] << '\n'
              << x;
    }
  catch(std::exception& x) {
    std::cerr << "Caught a std::exception in "
              << argv[0] << '\n'
              << x.what();
  }
  catch(...) {
    std::cerr << "Caught an unknown exception in "
              << argv[0];
  }
  return rc;
}
