
// $Id: ConsumerID.h,v 1.4 2011/03/07 15:31:31 mommsen Exp $
/// @file: ConsumerID.h 

#ifndef EventFilter_StorageManager_ConsumerID_h
#define EventFilter_StorageManager_ConsumerID_h

#include <cstddef>
#include <iostream>

#include "EventFilter/StorageManager/interface/EnquingPolicyTag.h"

namespace stor {

  /**
   * Uniquely identifies a consumer.
   *
   * $Author: mommsen $
   * $Revision: 1.4 $
   * $Date: 2011/03/07 15:31:31 $
   */

  struct ConsumerID
  {
    unsigned int value;

    explicit ConsumerID(unsigned int id = 0) : value(id) { }


    /**
       Return whether or not *this is a valid ConsumerID.
    */

    bool isValid() const { return value != 0; }

    /**
       operator< induces a strict weak ordering, so that ConsumerID can
       be used as a key in std::map.
    */
    bool operator< (ConsumerID other) const { return value < other.value; }

    /**
       operator== performs the expected equality test.
    */
    bool operator==(ConsumerID other) const { return value == other.value; }

    /**
       operator!= is the negation of operator==.
    */
    bool operator!=(ConsumerID other) const { return value != other.value; }

    /**
       operator++()  [preincrement] increments the given ConsumerID
       value, and returns the updated value.
     */
    ConsumerID& operator++() { ++value; return *this; }

    /**
       operator++(int) [postincrement] returns a copy of the current
       value of ConsumerID, and and increments *this.
     */
    ConsumerID operator++(int) {ConsumerID ret(*this); ++value; return ret;}
  };


  
  inline
  std::ostream&
  operator<< ( std::ostream& os, ConsumerID id)
  {
    return os << id.value;
  }

} // namespace stor

#endif // EventFilter_StorageManager_ConsumerID_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
