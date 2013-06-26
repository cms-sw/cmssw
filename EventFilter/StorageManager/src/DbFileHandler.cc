// $Id: DbFileHandler.cc,v 1.11 2012/04/04 12:17:00 mommsen Exp $
/// @file: DbFileHandler.cc

#include "EventFilter/StorageManager/interface/DbFileHandler.h"
#include "EventFilter/StorageManager/interface/Exception.h"

#include <iomanip>


namespace stor {
  
  DbFileHandler::DbFileHandler() :
  runNumber_(0)
  {}
  
  
  void DbFileHandler::writeOld(const utils::TimePoint_t& timestamp, const std::string& str)
  {
    std::ofstream outputFile;
    openFile(outputFile, timestamp);
    outputFile << str.c_str();
    outputFile << std::endl;
    outputFile.close();
  }
  
  
  void DbFileHandler::write(const std::string& str)
  {
    const utils::TimePoint_t timestamp = utils::getCurrentTime();
    
    std::ofstream outputFile;
    openFile(outputFile, timestamp);
    addReportHeader(outputFile, timestamp);
    outputFile << str.c_str();
    outputFile << std::endl;
    outputFile.close();
  }
  
  
  void DbFileHandler::configure(const unsigned int runNumber, const DiskWritingParams& params)
  { 
    dwParams_ = params;
    runNumber_ = runNumber;
    
    write("BoR");
  }
  
  
  void DbFileHandler::openFile
  (
    std::ofstream& outputFile,
    const utils::TimePoint_t& timestamp
  ) const
  {
    utils::checkDirectory(dwParams_.dbFilePath_);
    
    std::ostringstream dbfilename;
    dbfilename
      << dwParams_.dbFilePath_
        << "/"
        << utils::dateStamp(timestamp)
        << "-" << dwParams_.hostName_
        << "-" << dwParams_.smInstanceString_
        << ".log";
    
    outputFile.open( dbfilename.str().c_str(), std::ios_base::ate | std::ios_base::out | std::ios_base::app );
    if (! outputFile.is_open() )
    {
      std::ostringstream msg;
      msg << "Failed to open db log file " << dbfilename.str();
      XCEPT_RAISE(stor::exception::DiskWriting, msg.str());
    }
  }
  
  
  void DbFileHandler::addReportHeader
  (
    std::ostream& msg,
    const utils::TimePoint_t& timestamp
  ) const
  {
    msg << "Timestamp:" << utils::secondsSinceEpoch(timestamp)
      << "\trun:" << runNumber_
      << "\thost:" << dwParams_.hostName_
      << "\tinstance:" << dwParams_.smInstanceString_
      << "\t";
  }
  
} // namespace stor


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
