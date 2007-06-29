#ifndef StreamerStatService_h
#define StreamerStatService_h

// $Id: StreamerStatService.h,v 1.4 2006/10/11 14:39:32 klute Exp $

#include<string>
#include<vector>
#include<iosfwd>

namespace {
std::string itoa(int i){
        char temp[20];
        sprintf(temp,"%d",i);
        return((std::string)temp);
}

void tokenize(const std::string& str,
                      std::vector<std::string>& tokens,
                      const std::string& delimiters = " ") {
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

    while (std::string::npos != pos || std::string::npos != lastPos) {
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
       std::string streamer_;
       std::string dataFile_;
       std::string indexFile_;
       uint32 fileSize_;
       uint32 eventCount_;
       std::string startDate_;
       std::string startTime_;
       std::string endDate_;
       std::string endTime_;
   };

 //Class that reads a Stat file record by record (StatSummary) and provide info to user-layer.

 class StreamerStatReadService
  {
    public:
     explicit StreamerStatReadService(std::string statFileName);
     bool next();
     StatSummary& getRecord() { return currentRecord_; }

    private:
     StatSummary currentRecord_;
     std::string statFileName_; 
     std::auto_ptr<std::ifstream> statFile_;

  };//end of class   

//Class to deal with writting Stat 
class StreamerStatWriteService
   {

   public:
       explicit StreamerStatWriteService(uint32 run, std::string streamer, std::string dataFile, std::string indexFile, std::string statFileName);
       ~StreamerStatWriteService();
       void incrementEventCount();
       void advanceFileSize(uint32 increment);
       void setFileSize(uint32);
       void setEventCount(uint32);
       void setRunNumber(uint32);
       void writeStat();

   private:

       std::string getCurrentDate();
       std::string getCurrentTime();

       StatSummary summary_;
       std::string statFileName_;
   };

} //end-of-namespace

#endif
