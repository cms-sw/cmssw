
#include <iostream>
#include <string>
#include <cstdlib>
#include <cerrno>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace std;

int runme(const string& shell, const string& script)
{
  pid_t pid;
  int status=0;

  if((pid=fork())<0)
    {
      cerr << "fork failed, to run " << script << endl;
      return -1;
    }

  if(pid==0) // child
    {
      execlp(shell.c_str(),"sh","-c",script.c_str(),0);
      _exit(127);
    }
  else // parent
    {
      while(waitpid(pid,&status,0)<0)
	{
	  if(errno!=EINTR)
	    {
	      status=-1;
	      break;
	    }
	}
    }

  return status;
}

int main(int argc, char* argv[])
{
  if(argc<3)
    {
      cerr << "Usage: " << argv[0] << " shell script1 script2 ... scriptN"
	   << "\n\n  where shell is the path+shell e.g. /bin/bash "
	   << "to use to run the scripts"
	   <<  endl;
      return -1;
    }

  cout << get_current_dir_name() << endl;
  
  string sh(argv[1]);
  const char* thisdir = getenv("THISDIR");
  const char* topdir  = getenv("SCRAMRT_LOCALRT");
  const char* arch    = getenv("SCRAM_ARCH");
  int rc=0;

  if(!thisdir)
    {
      cerr << "THISDIR is not defined" << endl;
      return -1;
    }
  if(!topdir)
    {
      cerr << "SCRAMRT_LOCALRT is not defined" << endl;
      return -1;
    }
  if(!arch)
    {
      cerr << "SCRAMRT_ARCH is not defined" << endl;
      return -1;
    }

  string testdir(topdir); testdir+="/"; testdir+=thisdir; testdir+="/test";
  string tmpdir(topdir); tmpdir+="/tmp/"; tmpdir+=arch;
  string testbin(topdir); testbin+="/test/"; testbin+=arch;

  if(setenv("LOCAL_TEST_DIR",testdir.c_str(),1)!=0)
    {
      cerr << "Could not set LOCAL_TEST_DIR to " << testdir << endl;
      return -1;
    }
  if(setenv("LOCAL_TMP_DIR",tmpdir.c_str(),1)!=0)
    {
      cerr << "Could not set LOCAL_TMP_DIR to " << tmpdir << endl;
      return -1;
    }
  if(setenv("LOCAL_TOP_DIR",topdir.c_str(),1)!=0)
    {
      cerr << "Could not set LOCAL_TOP_DIR to " << topdir << endl;
      return -1;
    }
  if(setenv("LOCAL_TEST_BIN",testbin.c_str(),1)!=0)
    {
      cerr << "Could not set LOCAL_TEST_BIN to " << testbin << endl;
      return -1;
    }

  testdir+="/";

  for(int i=2;i<argc && rc==0;++i)
    {
      rc = runme(sh,testdir+argv[i]);
    }

  cout << "status = " << rc << endl;
  return rc!=0?-1:0;
}
