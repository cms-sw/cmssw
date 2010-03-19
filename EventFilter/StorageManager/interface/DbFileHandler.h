// $Id: DbFileHandler.h,v 1.9 2010/02/01 14:08:49 mommsen Exp $
/// @file: DbFileHandler.h 

#ifndef StorageManager_DbFileHandler_h
#define StorageManager_DbFileHandler_h

#include "EventFilter/StorageManager/interface/Configuration.h"

#include "boost/shared_ptr.hpp"

#include <string>


namespace stor {

  /**
   * Handle the file used to pass information into SM database
   *
   * $Author: mommsen $
   * $Revision: 1.9 $
   * $Date: 2010/02/01 14:08:49 $
   */

  class DbFileHandler
  {
  public:
        
    DbFileHandler() {};
    
    ~DbFileHandler() {};

    /**
     * Configure the db file writer
     */
    void configure(const unsigned int runNumber, const DiskWritingParams&);

    /**
     * Write the string into the db file. Close the file after each write.
     */
    void write(const std::string&) const;

    
  private:
    
    /**
     * Return the name of the database file
     */
    const char* dbFileName() const;

    //Prevent copying of the DbFileHandler
    DbFileHandler(DbFileHandler const&);
    DbFileHandler& operator=(DbFileHandler const&);
    
    DiskWritingParams _dwParams;
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
