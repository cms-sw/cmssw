// $Id: TransitionRecord.h,v 1.2 2009/06/10 08:15:24 dshpakov Exp $
/// @file: TransitionRecord.h 

#ifndef StorageManager_TransitionRecord_h
#define StorageManager_TransitionRecord_h

#include <ostream>
#include <sys/time.h>
#include <string>


namespace stor {

  /**
   * A record of state machine transitions
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:24 $
   */
  
  class TransitionRecord
  {
    
  public:

    TransitionRecord( const std::string& state_name,
                      bool is_entry );

    const std::string& stateName() const { return _stateName; }
    bool isEntry() const { return _isEntry; }
    const struct timeval& timeStamp() const { return _timestamp; }

    friend std::ostream& operator << ( std::ostream&,
                                       const TransitionRecord& );

  private:

    std::string _stateName;
    bool _isEntry;
    struct timeval _timestamp;

  };
  
} // namespace stor

#endif // StorageManager_TransitionRecord_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
