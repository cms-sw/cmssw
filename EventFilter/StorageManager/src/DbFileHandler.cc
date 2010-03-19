// $Id: DbFileHandler.cc,v 1.15 2010/02/18 10:16:23 mommsen Exp $
/// @file: DbFileHandler.cc

#include <EventFilter/StorageManager/interface/DbFileHandler.h>

#include <fstream>
#include <iomanip>

using namespace stor;
using namespace std;


void DbFileHandler::write(const string& str) const
{
  ofstream of(dbFileName(), ios_base::ate | ios_base::out | ios_base::app );
  of << str.c_str();
  of.close();
}


void DbFileHandler::configure(const unsigned int runNumber, const DiskWritingParams& params)
{ 
  _dwParams = params;

  std::ostringstream str;
  str << "Timestamp:" << static_cast<int>(utils::getCurrentTime())
    << "\trun:" << runNumber
    << "\tBoR\n";
  write(str.str());
}



const char* DbFileHandler::dbFileName() const
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
  return dbfilename.str().c_str();
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
