// $Id: QueueID.h,v 1.2 2009/06/10 08:15:23 dshpakov Exp $
/// @file: QueueID.h 

#ifndef StorageManager_QueueID_h
#define StorageManager_QueueID_h

#include <cstddef>
#include <iostream>

#include "EventFilter/StorageManager/interface/EnquingPolicyTag.h"

namespace stor {

  /**
   * Uniquely identifies the consumer queues 
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:23 $
   */

  class QueueID
    {
    public:
      typedef std::size_t size_type;
      typedef enquing_policy::PolicyTag policy_type;
      /**
         A default-constructed QueueID is invalid; it can not be used
         to identify an actual queue.
       */
      QueueID();

      /**
         Create a QueueID used to identify a queue with enquing policy
         denoted by policy and with identifier index.
       */
      QueueID(policy_type policy, size_t index);

      /**
         Return the tag for the queing policy of *this.
       */
      policy_type policy() const;

      /**
         Return the index for this queue.
       */
      size_t index() const;

      /**
         Test  for validity  of  a QueueID.  Invalid  QueueIDs do  not
         represent the identity of any actual queue.
       */
      bool isValid() const;

      /**
         operator< induces a strict weak ordering, so that QueueID can
         be used as a key in std::map.
      */
      bool operator< (QueueID const& other) const;

      /**
         operator== returns true if both the policies and indices are
         equal.
      */
      bool operator== (QueueID const& other) const;

      /**
         operator!= is the negation of operator==.
      */
      bool operator!= (QueueID const& other) const;

    private:
      size_type   _index;
      policy_type _policy;

    };

  inline
  QueueID::QueueID() :
    _index(0),
    _policy(enquing_policy::Max)
  { }

  inline 
  QueueID::QueueID(policy_type policy, size_t index) :
    _index(index),
    _policy(policy)
  { }

  inline 
  QueueID::policy_type
  QueueID::policy() const
  {
    return _policy;
  }

  inline
  QueueID::size_type
  QueueID::index() const
  {
    return _index;
  }

  inline
  bool
  QueueID::isValid() const
  {
    return _policy != enquing_policy::Max;
  }

  inline
  bool
  QueueID::operator< (QueueID const& other) const
  {
    return _policy == other._policy
      ? _index < other._index
      : _policy < other._policy;
  }

  inline
  bool
  QueueID::operator== (QueueID const& other) const
  {
    return _policy == other._policy && _index == other._index;
  }

  inline
  bool
  QueueID::operator!= (QueueID const& other) const
  {
    return !( operator==(other));
  }

  inline
  std::ostream&
  operator<< ( std::ostream& os, const QueueID& queueId )
  {
    os << "policy: " << queueId.policy() << 
      "   index: " << queueId.index();
    return os;
  }

} // namespace stor

#endif // StorageManager_QueueID_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
