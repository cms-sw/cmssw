// $Id: DbFileHandler.h,v 1.5 2011/03/07 15:31:31 mommsen Exp $
/// @file: DbFileHandler.h 

#ifndef EventFilter_StorageManager_DbFileHandler_h
#define EventFilter_StorageManager_DbFileHandler_h

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
   * $Revision: 1.5 $
   * $Date: 2011/03/07 15:31:31 $
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
    void writeOld(const utils::TimePoint_t&, const std::string&);

    /**
     * Write the string into the db file and prefix it with the report header.
     * Close the file after each write.
     */
    void write(const std::string&);

    /**
     * Return the DiskWritingParams used to configure the DbFileHandler
     */
    const DiskWritingParams& getDiskWritingParams() const
    { return dwParams_; }


  private:
    
    void openFile(std::ofstream&, const utils::TimePoint_t&) const;

    void addReportHeader(std::ostream&, const utils::TimePoint_t&) const;

    //Prevent copying of the DbFileHandler
    DbFileHandler(DbFileHandler const&);
    DbFileHandler& operator=(DbFileHandler const&);
    
    DiskWritingParams dwParams_;
    unsigned int runNumber_;
  };

  typedef boost::shared_ptr<DbFileHandler> DbFileHandlerPtr;
  
} // stor namespace

#endif // EventFilter_StorageManager_DbFileHandler_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
