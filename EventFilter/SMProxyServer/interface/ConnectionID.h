// $Id: ConnectionID.h,v 1.2 2011/03/07 15:41:54 mommsen Exp $
/// @file: ConnectionID.h 

#ifndef EventFilter_StorageManager_ConnectionID_h
#define EventFilter_StorageManager_ConnectionID_h

#include <cstddef>
#include <iostream>

namespace smproxy {

  /**
   * Uniquely identifies an event server connection.
   *
   * $Author: mommsen $
   * $Revision: 1.2 $
   * $Date: 2011/03/07 15:41:54 $
   */

  struct ConnectionID
  {
    unsigned int value;

    explicit ConnectionID(unsigned int id = 0) : value(id) { }


    /**
       Return whether or not *this is a valid ConnectionID.
    */
    bool isValid() const { return value != 0; }

    /**
       operator< induces a strict weak ordering, so that ConnectionID can
       be used as a key in std::map.
    */
    bool operator< (ConnectionID other) const { return value < other.value; }

    /**
       operator== performs the expected equality test.
    */
    bool operator==(ConnectionID other) const { return value == other.value; }

    /**
       operator!= is the negation of operator==.
    */
    bool operator!=(ConnectionID other) const { return value != other.value; }

    /**
       operator++()  [preincrement] increments the given ConnectionID
       value, and returns the updated value.
     */
    ConnectionID& operator++() { ++value; return *this; }

    /**
       operator++(int) [postincrement] returns a copy of the current
       value of ConnectionID, and and increments *this.
     */
    ConnectionID operator++(int) {ConnectionID ret(*this); ++value; return ret;}
  };


  
  inline
  std::ostream&
  operator<< ( std::ostream& os, ConnectionID id)
  {
    return os << id.value;
  }

} // namespace smproxy

#endif // EventFilter_StorageManager_ConnectionID_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
