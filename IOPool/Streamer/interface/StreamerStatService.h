#ifndef StreamerStatService_h
#define StreamerStatService_h
/**
  Keeps stat for single stream now, will be changed to accomodate multiple streams later in future.
*/

#include<iostream>
#include<ctime>
#include<string>
#include<vector>
#include<fstream>

using namespace std;

namespace {
string itoa(int i){
        char temp[20];
        sprintf(temp,"%d",i);
        return((string)temp);
}

void tokenize(const string& str,
                      vector<string>& tokens,
                      const string& delimiters = " ") {
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    string::size_type pos     = str.find_first_of(delimiters, lastPos);

    while (string::npos != pos || string::npos != lastPos) {
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = str.find_first_not_of(delimiters, pos);
        pos = str.find_first_of(delimiters, lastPos);
    }
}

} 

namespace edm
{
 // Stucture that represents information in a Stat/Summary file.
 // Summary/Stat file schema looks like following
 // run:streamer:output file:index file:data file size:number of events:start_date:start_time:end_date:end_time 
typedef unsigned int  uint32;
  
 struct StatSummary
   {
       uint32 run_;
       string streamer_;
       string dataFile_;
       string indexFile_;
       uint32 fileSize_;
       uint32 eventCount_;
       string startDate_;
       string startTime_;
       string endDate_;
       string endTime_;
   };

 //Class that reads a Stat file record by record (StatSummary) and provide info to user-layer.

 class StreamerStatReadService
  {
    public:
     explicit StreamerStatReadService(string statFileName);
     bool next();
     StatSummary& getRecord() { return currentRecord_; }

    private:
     StatSummary currentRecord_;
     string statFileName_; 
     auto_ptr<ifstream> statFile_;

  };//end of class   

StreamerStatReadService::StreamerStatReadService(string statFileName):
  statFileName_(statFileName),
  statFile_(new ifstream(statFileName_.c_str()))
  {
      
  }

bool StreamerStatReadService::next()
  {
     string curr_line;
     curr_line.resize(1000);
     if (statFile_->good() )
     if (statFile_->getline(&curr_line[0], 1000) )
        {
          vector<string> tokens;
          tokenize(curr_line, tokens, ":");
          //if (tokens.size() != 10) { raise exception here summary file has less/more info then expected }      
          currentRecord_.run_ = atoi(tokens[0].c_str());
          currentRecord_.streamer_ = tokens[1];
          currentRecord_.dataFile_ = tokens[2];
          currentRecord_.indexFile_ = tokens[3];
          currentRecord_.fileSize_ = atoi(tokens[4].c_str());
          currentRecord_.eventCount_ = atoi(tokens[5].c_str());
          currentRecord_.startDate_ = tokens[6];
          currentRecord_.startTime_ = tokens[7];
          currentRecord_.endDate_ = tokens[8];
          currentRecord_.endTime_ = tokens[9];
          return true;
        }
      return false;
  }


//Class to deal with writting Stat 
class StreamerStatWriteService
   {

   public:
       explicit StreamerStatWriteService(uint32 run, string streamer, string dataFile, string indexFile, string statFileName);
       ~StreamerStatWriteService();
       void incrementEventCount();
       void advanceFileSize(uint32 increment);
       void writeStat();

   private:

       string getCurrentDate();
       string getCurrentTime();

       StatSummary summary_;
       string statFileName_;
       auto_ptr<ofstream> statFile_;

   };

} //end-of-namespace

#endif
