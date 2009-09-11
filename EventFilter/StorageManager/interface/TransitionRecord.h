// $Id: TransitionRecord.h,v 1.3 2009/07/20 13:06:11 mommsen Exp $
/// @file: TransitionRecord.h 

#ifndef StorageManager_TransitionRecord_h
#define StorageManager_TransitionRecord_h

#include <iosfwd>
#include <sys/time.h>
#include <string>

namespace stor {

  /**
   * A record of state machine transitions
   *
   * $Author: mommsen $
   * $Revision: 1.3 $
   * $Date: 2009/07/20 13:06:11 $
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
  
  std::ostream& operator << ( std::ostream&, const TransitionRecord& );

} // namespace stor

#endif // StorageManager_TransitionRecord_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
