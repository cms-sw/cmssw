#include "IOPool/Streamer/interface/StreamerStatService.h"

namespace edm
{
StreamerStatWriteService::StreamerStatWriteService(uint32 run, string streamer, string dataFile, string indexFile, string statFileName):
  statFileName_(statFileName),
  statFile_(new ofstream(statFileName_.c_str(), ios_base::ate | ios_base::out | ios_base::app ))
  {
  summary_.run_=run;
  summary_.streamer_=streamer;
  summary_.dataFile_=dataFile;
  summary_.indexFile_=indexFile;
  summary_.fileSize_=0;
  summary_.eventCount_=0;
  summary_.startDate_=getCurrentDate();
  summary_.startTime_=getCurrentTime();

  }

StreamerStatWriteService::~StreamerStatWriteService()
  {

  }

void StreamerStatWriteService::incrementEventCount()
  {
     summary_.eventCount_++;
  }

string StreamerStatWriteService::getCurrentDate()
  {
   time_t rawtime; 
   tm * ptm; 
   time ( &rawtime ); 
   ptm = gmtime ( &rawtime ); 
   return string(itoa(ptm->tm_mday)+"/"+itoa(ptm->tm_mon)+"/"+itoa(ptm->tm_year+1900));
  }

string StreamerStatWriteService::getCurrentTime()
  {
  time_t rawtime;
  tm * ptm;
  time ( &rawtime );
  ptm = gmtime ( &rawtime );
  return string(itoa(ptm->tm_hour)+":"+itoa(ptm->tm_min));
  }

void StreamerStatWriteService::advanceFileSize(uint32 increment)
  {
    summary_.fileSize_ += increment;
  }

void StreamerStatWriteService::writeStat()
  {
     summary_.endDate_ = getCurrentDate();
     summary_.endTime_ = getCurrentTime();

     string currentStat = itoa(summary_.run_)+":"+summary_.streamer_+":"+summary_.dataFile_+":"+summary_.indexFile_+":"+
                          itoa(summary_.fileSize_)+":"+itoa(summary_.eventCount_)+":"+summary_.startDate_+":"+
                          summary_.startTime_+":"+summary_.endDate_+":"+summary_.endTime_+"\n";
     statFile_->write((char*)&currentStat[0], currentStat.length());
  }
} //end-of-namespace

