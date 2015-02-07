#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/StreamerInputFile.h"
#include "CalibCalorimetry/EcalLaserSorting/interface/WatcherStreamFileReader.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <libgen.h>
#include <fstream>

//using namespace edm;
using namespace std;

//std::string WatcherStreamFileReader::fileName_;


#if !defined(__linux__) && !(defined(__APPLE__) && __DARWIN_C_LEVEL >= 200809L)
/* getline implementation is copied from glibc. */

#ifndef SIZE_MAX
# define SIZE_MAX ((size_t) -1)
#endif
#ifndef SSIZE_MAX
# define SSIZE_MAX ((ssize_t) (SIZE_MAX / 2))
#endif
namespace {
ssize_t getline (char **lineptr, size_t *n, FILE *fp)
{
    ssize_t result = -1;
    size_t cur_len = 0;

    if (lineptr == NULL || n == NULL || fp == NULL)
    {
        errno = EINVAL;
        return -1;
   }

    if (*lineptr == NULL || *n == 0)
    {
        *n = 120;
        *lineptr = (char *) malloc (*n);
        if (*lineptr == NULL)
        {
            result = -1;
            goto end;
        }
    }

    for (;;)
    {
        int i;

        i = getc (fp);
        if (i == EOF)
        {
            result = -1;
            break;
        }

        /* Make enough space for len+1 (for final NUL) bytes.  */
        if (cur_len + 1 >= *n)
        {
            size_t needed_max =
                SSIZE_MAX < SIZE_MAX ? (size_t) SSIZE_MAX + 1 : SIZE_MAX;
            size_t needed = 2 * *n + 1;   /* Be generous. */
            char *new_lineptr;

            if (needed_max < needed)
                needed = needed_max;
            if (cur_len + 1 >= needed)
            {
                result = -1;
                goto end;
            }

            new_lineptr = (char *) realloc (*lineptr, needed);
            if (new_lineptr == NULL)
            {
                result = -1;
                goto end;
            }

            *lineptr = new_lineptr;
            *n = needed;
        }

        (*lineptr)[cur_len] = i;
        cur_len++;

        if (i == '\n')
            break;
    }
    (*lineptr)[cur_len] = '\0';
    result = cur_len ? (ssize_t) cur_len : result;

end:
    return result;
}
}
#endif

static std::string now(){
  struct timeval t;
  gettimeofday(&t, 0);
 
  char buf[256];
  strftime(buf, sizeof(buf), "%F %R %S s", localtime(&t.tv_sec));
  buf[sizeof(buf)-1] = 0;

  stringstream buf2;
  buf2 << buf << " " << ((t.tv_usec+500)/1000)  << " ms";

  return buf2.str();
}

WatcherStreamFileReader::WatcherStreamFileReader(edm::ParameterSet const& pset):
  inputDir_(pset.getParameter<std::string>("inputDir")),
  filePatterns_(pset.getParameter<std::vector<std::string> >("filePatterns")),
  inprocessDir_(pset.getParameter<std::string>("inprocessDir")),
  processedDir_(pset.getParameter<std::string>("processedDir")),
  corruptedDir_(pset.getParameter<std::string>("corruptedDir")),
  tokenFile_(pset.getUntrackedParameter<std::string>("tokenFile",
						     "watcherSourceToken")),
  timeOut_(pset.getParameter<int>("timeOutInSec")),
  end_(false),
  verbosity_(pset.getUntrackedParameter<int>("verbosity", 0)){
  struct stat buf;
  if(stat(tokenFile_.c_str(), &buf)){
    FILE* f = fopen(tokenFile_.c_str(), "w");
    if(f){
      fclose(f);
    } else{
      throw cms::Exception("WatcherSource") << "Failed to create token file.";
    }
  }
  vector<string> dirs;
  dirs.push_back(inprocessDir_);
  dirs.push_back(processedDir_);
  dirs.push_back(corruptedDir_);
  
  for(unsigned i = 0; i < dirs.size(); ++i){
    const string& dir = dirs[i];
    struct stat fileStat;
    if(0==stat(dir.c_str(), &fileStat)){
      if(!S_ISDIR(fileStat.st_mode)){
	throw cms::Exception("[WatcherSource]")
	  << "File " << dir << " exists but is not a directory "
	  << " as expected.";
      }
    } else {//directory does not exists, let's try to create it
      if(0!=mkdir(dir.c_str(), 0755)){
	throw cms::Exception("[WatcherSource]")
	  << "Failed to create directory " << dir
	  << " for writing data.";
      }
    }
  }

    std::stringstream fileListCmdBuf;
    fileListCmdBuf.str("");
    //    fileListCmdBuf << "/bin/ls -rt " << inputDir_ << " | egrep '(";
    //by default ls will sort the file alphabetically which will results
    //in ordering the files in increasing LB number, which is the desired
    //order.
    //    fileListCmdBuf << "/bin/ls " << inputDir_ << " | egrep '(";
    fileListCmdBuf << "/bin/find " << inputDir_ << " -maxdepth 2 -print | egrep '(";
    //TODO: validate patternDir (see ;, &&, ||) and escape special character
    if(filePatterns_.size()==0) throw cms::Exception("WacherSource", "filePatterns parameter is empty");
    char curDir[PATH_MAX>0?PATH_MAX:4096];
    if(getcwd(curDir, sizeof(curDir))==0){
      throw cms::Exception("WatcherSource")
	<< "Failed to retreived working directory path: "
	<< strerror(errno);
    }
    curDir_ = curDir;
    
    for(unsigned i = 0 ; i < filePatterns_.size(); ++i){
      if(i>0) fileListCmdBuf << "|";
      //     if(filePatterns_[i].size()>0 && filePatterns_[0] != "/"){//relative path
      //       fileListCmdBuf << curDir << "/";
      //     }
      fileListCmdBuf << filePatterns_[i];
    }
    fileListCmdBuf << ")' | sort";

    fileListCmd_ = fileListCmdBuf.str();
    
    cout << "[WatcherSource " << now() << "]" 
	 << " Command to retrieve input files: "
	 << fileListCmd_ << "\n";

}

WatcherStreamFileReader::~WatcherStreamFileReader(){
}

const bool WatcherStreamFileReader::newHeader() {
  edm::StreamerInputFile* inputFile = getInputFile();
  return inputFile?inputFile->newHeader():false;
}

const InitMsgView* WatcherStreamFileReader::getHeader(){

  edm::StreamerInputFile* inputFile = getInputFile();

  //TODO: shall better send an exception...
  if(inputFile==0){
    throw cms::Exception("WatcherSource") << "No input file found.";
  }
  
  const InitMsgView* header = inputFile->startMessage();
  
  if(header->code() != Header::INIT) //INIT Msg
    throw cms::Exception("readHeader","WatcherStreamFileReader")
      << "received wrong message type: expected INIT, got "
      << header->code() << "\n";
    
  return header;
}
  
const EventMsgView* WatcherStreamFileReader::getNextEvent(){
  if(end_){ closeFile(); return 0;}
  
  edm::StreamerInputFile* inputFile;

  //go to next input file, till no new event is found
  while((inputFile=getInputFile())!=0
	&& inputFile->next()==0){
    closeFile();
  }

  return inputFile==0?0:inputFile->currentRecord();
}

edm::StreamerInputFile* WatcherStreamFileReader::getInputFile(){
  char* lineptr = 0;
  size_t n = 0;

  struct stat buf;
  
  if(stat(tokenFile_.c_str(), &buf)!=0){ 
    end_ = true; 
  }
  
  bool waiting = false;
  static bool firstWait = true;
  timeval waitStart;
  //if no cached input file, look for new files until one is found:
  if(!end_ && streamerInputFile_.get()==0){
    fileName_.assign("");
    
    //check if we have file in the queue, if not look for new files:
    while(filesInQueue_.size()==0){
      if(stat(tokenFile_.c_str(), &buf)!=0){ 
	end_ = true; 
	break;
      }
      FILE* s = popen(fileListCmd_.c_str(), "r");
      if(s==0){
	throw cms::Exception("WatcherSource")
	  << "Failed to retrieve list of input file: " << strerror(errno);
      }
      
      ssize_t len;
      while(!feof(s)){
	if((len=getline(&lineptr, &n, s))>0){
	  //remove end-of-line character:
	  lineptr[len-1] = 0;
	  string fileName;
	  if(lineptr[0] != '/'){
	    if(inputDir_.size()>0 && inputDir_[0] != '/'){//relative path
	      fileName.assign(curDir_);
	      fileName.append("/");
	      fileName.append(inputDir_);
	    } else{
	      fileName.assign(inputDir_);
	    }
	    fileName.append("/");
	  }
	  fileName.append(lineptr);
	  filesInQueue_.push_back(fileName);
	  if(verbosity_) cout << "[WatcherSource " << now() << "]" 
			      << " File to process: '"
			      << fileName << "'\n";
	}
      }
      while(!feof(s)) fgetc(s);
      pclose(s);
      if(filesInQueue_.size()==0){
	if(!waiting){
	  cout << "[WatcherSource " << now() << "]" 
	       << " No file found. Waiting for new file...\n";
	  cout << flush;
	  waiting = true;
	  gettimeofday(&waitStart, 0);
	} else if(!firstWait){
	  timeval t;
	  gettimeofday(&t, 0);
	  float dt = (t.tv_sec-waitStart.tv_sec) * 1.
	    + (t.tv_usec-waitStart.tv_usec) * 1.e-6;
	  if((timeOut_ >= 0) && (dt > timeOut_)){
	    cout << "[WatcherSource " << now() << "]"
		 << " Having waited for new file for " << (int)dt << " sec. "
		 << "Timeout exceeded. Exits.\n";
	    //remove(tokenFile_.c_str()); //we do not delete the token, otherwise sorting process on the monitoring farm will not be restarted by the runloop.sh script.
	    end_ = true;
	    break;
	  }
	}
      }
      sleep(1);
    } //end of file queue update
    firstWait = false;
    free(lineptr); lineptr=0;
    
    while(streamerInputFile_.get()==0 && !filesInQueue_.empty()){

      fileName_ = filesInQueue_.front();
      filesInQueue_.pop_front();
      int fd = open(fileName_.c_str(), 0);
      if(fd!=0){
	struct stat buf;
	off_t size = -1;
	//check that file transfer is finished, by monitoring its size:
	time_t t = time(0);
	for(;;){
	  fstat(fd, &buf);
	  if(verbosity_) cout << "file size: " << buf.st_size << ", prev size: "  << size << "\n";
	  if(buf.st_size==size) break; else size = buf.st_size;
	  if(difftime(t,buf.st_mtime)>60) break; //file older then 1 min=> tansfer must be finished
	  sleep(1);
	}

	if(fd!=0 && buf.st_size == 0){//file is empty. streamer reader 
	  //                   does not like empty file=> skip it
	  stringstream c;
	  c << "/bin/mv -f \"" << fileName_ << "\" \"" << corruptedDir_
	    << "/.\"";
	  if(verbosity_) cout << "[WatcherSource " << now() << "]" 
			      << " Excuting "
			      << c.str() << "\n"; 
	  int i = system(c.str().c_str());
	  if(i!=0){
	    //throw cms::Exception("WatcherSource")
	    cout << "[WatcherSource " << now() << "] "
		 << "Failed to move empty file '" << fileName_ << "'"
		 << " to corrupted directory '" << corruptedDir_ << "'\n";
	  }
	  continue;
	}
	
	close(fd);

	vector<char> buf1(fileName_.size()+1);
	copy(fileName_.begin(), fileName_.end(), buf1.begin());
	buf1[buf1.size()-1] = 0;
	
	vector<char> buf2(fileName_.size()+1);
	copy(fileName_.begin(), fileName_.end(), buf2.begin());
	buf2[buf1.size()-1] = 0;
	
	string dirnam(dirname(&buf1[0]));
	string filenam(basename(&buf2[0]));
	
	string dest  = inprocessDir_ + "/" + filenam;
	
	if(verbosity_) cout << "[WatcherSource " << now() << "]" 
			    << " Moving file "
			    << fileName_ << " to " << dest << "\n";
	
	stringstream c;
	c << "/bin/mv -f \"" << fileName_ << "\" \"" << dest
	  << "/.\"";
	

	if(0!=rename(fileName_.c_str(), dest.c_str())){
	  //if(0!=system(c.str().c_str())){
	  throw cms::Exception("WatcherSource")
	    << "Failed to move file '" << fileName_ << "' "
	    << "to processing directory " << inprocessDir_
	    << ": " << strerror(errno);
	}
	
	fileName_ = dest;

	cout << "[WatcherSource " << now() << "]" 
	     << " Opening file " << fileName_ << "\n" << flush;
	streamerInputFile_
	  = auto_ptr<edm::StreamerInputFile>(new edm::StreamerInputFile(fileName_));

	ofstream f(".watcherfile");
	f << fileName_;	
      } else{
	cout << "[WatcherSource " << now() << "]" 
	     << " Failed to open file " << fileName_ << endl;
      }
    } //loop on file queue to find one file which opening succeeded
  }
  return streamerInputFile_.get();
}

void WatcherStreamFileReader::closeFile(){
  if(streamerInputFile_.get()==0) return;
  //delete the streamer input file:
  streamerInputFile_.reset();
  stringstream cmd;
  //TODO: validation of processDir
  cmd << "/bin/mv -f \"" << fileName_ << "\" \"" << processedDir_ << "/.\"";
  if(verbosity_) cout << "[WatcherSource " << now() << "]" 
		      << " Excuting " << cmd.str() << "\n"; 
  int i = system(cmd.str().c_str());
  if(i!=0){
    throw cms::Exception("WatcherSource")
      << "Failed to move processed file '" << fileName_ << "'"
      << " to processed directory '" << processedDir_ << "'\n";
    //Stop further processing to prevent endless loop:
    end_ = true;
  }
  cout << flush;
}
