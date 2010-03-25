// $Id: DbFileHandler.cc,v 1.2 2010/03/19 17:33:54 mommsen Exp $
/// @file: DbFileHandler.cc

#include <EventFilter/StorageManager/interface/DbFileHandler.h>

#include <iomanip>

using namespace stor;
using namespace std;


DbFileHandler::DbFileHandler() :
_runNumber(0)
{}


void DbFileHandler::writeOld(const string& str)
{
  std::ofstream outputFile;
  openFile(outputFile);
  outputFile << str.c_str();
  outputFile.close();
}


void DbFileHandler::write(const string& str)
{
  std::ofstream outputFile;
  openFile(outputFile);
  addReportHeader(outputFile);
  outputFile << str.c_str();
  outputFile << endl;
  outputFile.close();
}


void DbFileHandler::configure(const unsigned int runNumber, const DiskWritingParams& params)
{ 
  _dwParams = params;
  _runNumber = runNumber;

  write("BoR");
}



void DbFileHandler::openFile(std::ofstream& outputFile) const
{
  time_t rawtime = time(0);
  tm * ptm;
  ptm = localtime(&rawtime);

  string dbPath(_dwParams._filePath+"/log");
  utils::checkDirectory(dbPath);

  ostringstream dbfilename;
  dbfilename << dbPath << "/"
             << setfill('0') << setw(4) << ptm->tm_year+1900
             << setfill('0') << setw(2) << ptm->tm_mon+1
             << setfill('0') << setw(2) << ptm->tm_mday
             << "-" << _dwParams._hostName
             << "-" << _dwParams._smInstanceString
             << ".log";

  outputFile.open( dbfilename.str().c_str(), ios_base::ate | ios_base::out | ios_base::app );
}


void DbFileHandler::addReportHeader(std::ostream& msg) const
{
  msg << "Timestamp:" << static_cast<int>(utils::getCurrentTime())
    << "\trun:" << _runNumber << "\t";
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
