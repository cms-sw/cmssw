#include "IOPool/Streamer/interface/MsgTools.h"
#include "CalibCalorimetry/EcalLaserSorting/interface/WatcherStreamFileReader.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <libgen.h>

using namespace edm;
using namespace std;

WatcherStreamFileReader::WatcherStreamFileReader(edm::ParameterSet const& pset):
  inputDir_(pset.getParameter<std::string>("inputDir")),
  filePatterns_(pset.getParameter<std::vector<std::string> >("filePatterns")),
  inprocessDir_(pset.getParameter<std::string>("inprocessDir")),
  processedDir_(pset.getParameter<std::string>("processedDir")),
  corruptedDir_(pset.getParameter<std::string>("corruptedDir")),
  tokenFile_(pset.getUntrackedParameter<std::string>("tokenFile",
						     "watcherSourceToken")),
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
}

WatcherStreamFileReader::~WatcherStreamFileReader(){
}

const bool WatcherStreamFileReader::newHeader() {
  StreamerInputFile* inputFile = getInputFile();
  return inputFile?inputFile->newHeader():false;
}

const InitMsgView* WatcherStreamFileReader::getHeader(){

  StreamerInputFile* inputFile = getInputFile();

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
  
  StreamerInputFile* inputFile;

  //go to next input file, till no new event is found
  while((inputFile=getInputFile())!=0
	&& inputFile->next()==0){
    closeFile();
  }

  return inputFile==0?0:inputFile->currentRecord();
}

StreamerInputFile* WatcherStreamFileReader::getInputFile(){
  char* lineptr = 0;
  size_t n = 0;
  static stringstream cmd;
  static bool cmdSet = false;
  static char curDir[PATH_MAX>0?PATH_MAX:4096];

  if(!cmdSet){
    cmd.str("");
    cmd << "/bin/ls -rt " << inputDir_ << " | egrep '(";
    //TODO: validate patternDir (see ;, &&, ||) and escape special character
    if(filePatterns_.size()==0) return 0;
    if(getcwd(curDir, sizeof(curDir))==0){
      throw cms::Exception("WatcherSource")
	<< "Failed to retreived working directory path: "
	<< strerror(errno);
    }
    
    for(unsigned i = 0 ; i < filePatterns_.size(); ++i){
      if(i>0) cmd << "|";
//     if(filePatterns_[i].size()>0 && filePatterns_[0] != "/"){//relative path
//       cmd << curDir << "/";
//     }
      cmd << filePatterns_[i];
    }
    cmd << ")'";
    
    cout << "[WatcherSource] Command to retrieve input files: "
	 << cmd.str() << "\n";
    cmdSet = true;
  }

  struct stat buf;
  
  if(stat(tokenFile_.c_str(), &buf)!=0){ 
      end_ = true; 
  }
  
  bool waitMess = true;
  //if no cached input file, look for new files until one is found:
  if(!end_ && streamerInputFile_.get()==0){
    fileName_.assign("");
    
    //check if we have file in the queue, if not look for new files:
    while(filesInQueue_.size()==0){
      if(stat(tokenFile_.c_str(), &buf)!=0){ 
	end_ = true; 
	break;
      }
      FILE* s = popen(cmd.str().c_str(), "r");
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
	  if(inputDir_.size()>0 && inputDir_ != "/"){//relative path
	    fileName.assign(curDir);
	    fileName.append("/");
	    fileName.append(inputDir_);
	  } else{
	    fileName.assign(inputDir_);
	  }
	  fileName.append("/");
	  fileName.append(lineptr);
	  filesInQueue_.push_back(fileName);
	  if(verbosity_) cout << "[WatcherSource] File to process: '"
			      << fileName << "'\n";
	}
      }
      while(!feof(s)) fgetc(s);
      pclose(s);
      if(filesInQueue_.size()==0){
	if(waitMess){
	  cout << "[WatcherSource] No file found. Waiting for new file...\n";
	  cout << flush;
	  waitMess = false;
	}
      }
      sleep(1);
    } //end of file queue update
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
	  if(verbosity_) cout << "[WatcherSource] Excuting "
			      << c.str() << "\n"; 
	  int i = system(c.str().c_str());
	  if(i!=0){
	    //throw cms::Exception("WatcherSource")
	    cout << "[WatcherSource]"
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
	
	if(verbosity_) cout << "[WatcherSource] Moving file "
			    << fileName_ << " to " << dest << "\n";
	
	if(0!=rename(fileName_.c_str(), dest.c_str())){
	  throw cms::Exception("WatcherSource")
	  << "Failed to move file '" << fileName_ << "' "
	  << "to processing directory " << inprocessDir_
	  << ": " << strerror(errno);
	}
	
	fileName_ = dest;

	cout << "[WatcherSource] Opening file " << fileName_ << "\n" << flush;
	streamerInputFile_
	  = auto_ptr<StreamerInputFile>(new StreamerInputFile(fileName_));
      } else{
	cout << "[WatcherSource] Failed to open file " << fileName_ << endl;
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
  if(verbosity_) cout << "[WatcherSource] Excuting " << cmd.str() << "\n"; 
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
