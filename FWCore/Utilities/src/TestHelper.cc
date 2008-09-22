//------------------------------------------------------------
// $Id: TestHelper.cc,v 1.7 2007/06/14 02:01:01 wmtan Exp $
//------------------------------------------------------------
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include <sys/wait.h>
#include <unistd.h>

#include "boost/filesystem/convenience.hpp"
#include "boost/filesystem/path.hpp"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TestHelper.h"

namespace bf=boost::filesystem;

//man pages for environ say you must declare it as such
extern char** environ;

int run_script(const std::string& shell, const std::string& script)
{
  pid_t pid;
  int status=0;

  if ((pid=fork())<0)
    {
      std::cerr << "fork failed, to run " << script << std::endl;;
      return -1;
    }

  if (pid==0) // child
    {
      execlp(shell.c_str(), "sh", "-c", script.c_str(), 0);
      std::cerr <<"child failed becuase '"<<strerror(errno)<<"'\n";
      _exit(127); // signal parent and children processes
    }
  else // parent
    {
      while(waitpid(pid,&status,0)<0)
	{
	  if (errno!=EINTR)
	    {
              std::cerr <<"child process failed "<<strerror(errno)<<"\n";
	      status=-1;
	      break;
	    } else {
              if( WIFSIGNALED(status) ) {
                std::cerr << "child existed because of a signal "<<WTERMSIG(status)<<"\n";
              }
            }
	}
    if( WIFSIGNALED(status) ) {
      std::cerr << "child existed because of a signal "<<WTERMSIG(status)<<"\n";
    }
    if(WIFEXITED(status)) {
    }
    
    }
  return status;
}



int do_work(int argc, char* argv[])
{
  bf::path currentPath(bf::initial_path().string(), bf::no_check);
  
  if (argc<4)
    {
      std::cout << "Usage: " << argv[0] << " shell subdir script1 script2 ... scriptN\n\n"
		<< "where shell is the path+shell (e.g., /bin/bash) intended to run the scripts\n"
		<< "and subdir is the subsystem/package/subdir in which the scripts are found\n"
		<< "(e.g., FWCore/Utilities/test)\n"
		<< std::endl;

      std::cout << "Current directory is: " << currentPath.native_directory_string() << '\n';
      std::cout << "Current environment:\n";
      std::cout << "---------------------\n";
      for (int i = 0; environ[i] != 0; ++i) std::cout << environ[i] << '\n';
      std::cout << "---------------------\n";
      std::cout << "Executable name: " << argv[0] << '\n';
      return -1;
    }

  for (int i = 0; i < argc; ++i)
    {
      std::cout << "argument " << i << ": " << argv[i] << '\n';
    }

  
  std::string shell(argv[1]);
  std::cerr << "shell is: " << shell << '\n';

  std::cout << "Current directory is: " << currentPath.native_directory_string() << '\n';
  // It is unclear about which of these environment variables should
  // be used.
  const char* topdir  = getenv("SCRAMRT_LOCALRT");
  if (!topdir) topdir = getenv("LOCALRT");

  const char* arch    = getenv("SCRAM_ARCH");

  if ( !arch )
    {
      // Try to synthesize SCRAM_ARCH value.
      bf::path exepath(argv[0], bf::no_check);
      std::string maybe_arch = exepath.branch_path().leaf();
      if (setenv("SCRAM_ARCH", maybe_arch.c_str(), 1) != 0)
	{
	  std::cerr << "SCRAM_ARCH not set and attempt to set it failed\n";
	  return -1;
	}
    }

  int rc=0;

  if (!topdir)
    {
      std::cout << "Neither SCRAMRT_LOCALRT nor LOCALRT is not defined" << std::endl;;
      return -1;
    }


  std::string testdir(topdir); testdir += "/src/"; testdir += argv[2];
  std::string tmpdir(topdir);  tmpdir+="/tmp/";   tmpdir+=arch;
  std::string testbin(topdir); testbin+="/test/"; testbin+=arch;

  std::cout << "topdir is: " << topdir << '\n';
  std::cout << "testdir is: " << testdir << '\n';
  std::cout << "tmpdir is: " << tmpdir << '\n';
  std::cout << "testbin is: " << testbin << '\n';


  if (setenv("LOCAL_TEST_DIR",testdir.c_str(),1)!=0)
    {
      std::cerr << "Could not set LOCAL_TEST_DIR to " << testdir << std::endl;;
      return -1;
    }
  if (setenv("LOCAL_TMP_DIR",tmpdir.c_str(),1)!=0)
    {
      std::cerr << "Could not set LOCAL_TMP_DIR to " << tmpdir << std::endl;;
      return -1;
    }
  if (setenv("LOCAL_TOP_DIR",topdir,1)!=0)
    {
      std::cerr << "Could not set LOCAL_TOP_DIR to " << topdir << std::endl;;
      return -1;
    }
  if (setenv("LOCAL_TEST_BIN",testbin.c_str(),1)!=0)
    {
      std::cerr << "Could not set LOCAL_TEST_BIN to " << testbin << std::endl;;
      return -1;
    }

  testdir+="/";

  for(int i=3; i<argc && rc==0; ++i)
    {
      std::string scriptname(testdir);
      scriptname += argv[i];
      std::cout << "Running script: " << scriptname << std::endl;
      rc = run_script(shell, scriptname);
    }

  std::cout << "status = " << rc << std::endl;;
  return rc == 0 ? 0 : -1;
}

int ptomaine(int argc, char* argv[])
{
  int rc = 1;
  try
    {
      rc = do_work(argc, argv);
    }
  catch ( edm::Exception& x )
    {
      std::cerr << "Caught an edm::Exception in "
		<< argv[0] << '\n'
		<< x;
    }
  catch ( cms::Exception& x )
    {
      std::cerr << "Caught a cms::Exception in "
		<< argv[0] << '\n'
		<< x;
    }
  catch ( std::exception& x )
    {
      std::cerr << "Caught a std::exception in "
		<< argv[0] << '\n'
		<< x.what();
    }
  catch (...)
    {
      std::cerr << "Caught an unknown exception in "
		<< argv[0];
    }
  return rc;
}
