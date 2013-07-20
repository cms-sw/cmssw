// $Id: TransitionRecord.h,v 1.5 2011/03/07 15:31:32 mommsen Exp $
/// @file: TransitionRecord.h 

#ifndef EventFilter_StorageManager_TransitionRecord_h
#define EventFilter_StorageManager_TransitionRecord_h

#include <iosfwd>
#include <sys/time.h>
#include <string>

namespace stor {

  /**
   * A record of state machine transitions
   *
   * $Author: mommsen $
   * $Revision: 1.5 $
   * $Date: 2011/03/07 15:31:32 $
   */
  
  class TransitionRecord
  {
    
  public:

    TransitionRecord
    (
      const std::string& stateName,
      bool isEntry
    );

    const std::string& stateName() const { return stateName_; }
    bool isEntry() const { return isEntry_; }
    const struct timeval& timeStamp() const { return timestamp_; }

    friend std::ostream& operator << ( std::ostream&,
                                       const TransitionRecord& );

  private:

    std::string stateName_;
    bool isEntry_;
    struct timeval timestamp_;

  };
  
  std::ostream& operator << ( std::ostream&, const TransitionRecord& );

} // namespace stor

#endif // EventFilter_StorageManager_TransitionRecord_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
