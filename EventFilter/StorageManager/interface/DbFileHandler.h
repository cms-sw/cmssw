// $Id: DbFileHandler.h,v 1.2 2010/03/19 17:34:06 mommsen Exp $
/// @file: DbFileHandler.h 

#ifndef StorageManager_DbFileHandler_h
#define StorageManager_DbFileHandler_h

#include "EventFilter/StorageManager/interface/Configuration.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/mutex.hpp"

#include <fstream>
#include <string>


namespace stor {

  /**
   * Handle the file used to pass information into SM database
   *
   * $Author: mommsen $
   * $Revision: 1.2 $
   * $Date: 2010/03/19 17:34:06 $
   */

  class DbFileHandler
  {
  public:
        
    DbFileHandler();
    
    ~DbFileHandler() {};

    /**
     * Configure the db file writer
     */
    void configure(const unsigned int runNumber, const DiskWritingParams&);

    /**
     * Write the string into the db file. Close the file after each write.
     */
    void writeOld(const std::string&);

    /**
     * Write the string into the db file and prefix it with the report header.
     * Close the file after each write.
     */
    void write(const std::string&);

    
  private:
    
    void openFile(std::ofstream&) const;

    void addReportHeader(std::ostream&) const;

    //Prevent copying of the DbFileHandler
    DbFileHandler(DbFileHandler const&);
    DbFileHandler& operator=(DbFileHandler const&);
    
    DiskWritingParams _dwParams;
    unsigned int _runNumber;
  };

  typedef boost::shared_ptr<DbFileHandler> DbFileHandlerPtr;
  
} // stor namespace

#endif // StorageManager_DbFileHandler_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
