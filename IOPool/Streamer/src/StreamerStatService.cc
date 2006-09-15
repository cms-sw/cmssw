#include "IOPool/Streamer/interface/StreamerStatService.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm
{

StreamerStatReadService::StreamerStatReadService(std::string statFileName):
  statFileName_(statFileName),
  statFile_(new ifstream(statFileName_.c_str()))
  {
   if(!statFile_->is_open()) {
       throw cms::Exception("StreamerStatReadService","StreamerStatReadService")
       << "Error Opening Output File: "<< statFileName_ <<"\n";
   }

  }

bool StreamerStatReadService::next()
  {
     std::string curr_line;
     curr_line.resize(1000);
     if (statFile_->good() )
     if (statFile_->getline(&curr_line[0], 1000) )
        {
          std::vector<std::string> tokens;
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





StreamerStatWriteService::StreamerStatWriteService( uint32 run, std::string streamer, 
                                                    std::string dataFile, std::string indexFile, 
                                                    std::string statFileName):
  statFileName_(statFileName),
  statFile_(new ofstream(statFileName_.c_str(), ios_base::ate | ios_base::out | ios_base::app ))
  {
    if(!statFile_->is_open()) {
        throw cms::Exception("StreamerStatWriteService","StreamerStatWriteService")
        << "Error Opening Output File: "<<statFileName_<<"\n";
     }
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

std::string StreamerStatWriteService::getCurrentDate()
  {
   time_t rawtime; 
   tm * ptm; 
   time ( &rawtime ); 
   ptm = gmtime ( &rawtime ); 
   return std::string(itoa(ptm->tm_mday)+"/"+itoa(ptm->tm_mon)+"/"+itoa(ptm->tm_year+1900));
  }

std::string StreamerStatWriteService::getCurrentTime()
  {
  time_t rawtime;
  tm * ptm;
  time ( &rawtime );
  ptm = gmtime ( &rawtime );
  return std::string(itoa(ptm->tm_hour)+":"+itoa(ptm->tm_min));
  }

void StreamerStatWriteService::advanceFileSize(uint32 increment)
  {
    summary_.fileSize_ += increment;
  }

void StreamerStatWriteService::writeStat()
  {
     summary_.endDate_ = getCurrentDate();
     summary_.endTime_ = getCurrentTime();

     std::string currentStat = itoa(summary_.run_)+":"+summary_.streamer_+":"
                                             +summary_.dataFile_+":"
                                             +summary_.indexFile_+":"
                                             +itoa(summary_.fileSize_)
                                             +":"+itoa(summary_.eventCount_)+":"
                                             +summary_.startDate_+":"
                                             +summary_.startTime_+":"+summary_.endDate_
                                             +":"+summary_.endTime_+"\n";
     statFile_->write((char*)&currentStat[0], currentStat.length());
  }
} //end-of-namespace

