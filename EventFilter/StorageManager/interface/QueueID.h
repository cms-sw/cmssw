// $Id: QueueID.h,v 1.4 2011/03/07 15:31:32 mommsen Exp $
/// @file: QueueID.h 

#ifndef EventFilter_StorageManager_QueueID_h
#define EventFilter_StorageManager_QueueID_h

#include <cstddef>
#include <iostream>
#include <vector>

#include "EventFilter/StorageManager/interface/EnquingPolicyTag.h"

namespace stor {

  /**
   * Uniquely identifies the consumer queues 
   *
   * $Author: mommsen $
   * $Revision: 1.4 $
   * $Date: 2011/03/07 15:31:32 $
   */

  class QueueID
    {
    public:

      typedef enquing_policy::PolicyTag PolicyType;

      /**
         A default-constructed QueueID is invalid; it can not be used
         to identify an actual queue.
       */
      QueueID();

      /**
         Create a QueueID used to identify a queue with enquing policy
         denoted by policy and with identifier index.
       */
      QueueID(PolicyType policy, size_t index);

      /**
         Return the tag for the queing policy of *this.
       */
      PolicyType policy() const;

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
      size_t   index_;
      PolicyType policy_;

    };

  typedef std::vector<QueueID> QueueIDs;

  inline
  QueueID::QueueID() :
    index_(0),
    policy_(enquing_policy::Max)
  { }

  inline 
  QueueID::QueueID(PolicyType policy, size_t index) :
    index_(index),
    policy_(policy)
  { }

  inline 
  QueueID::PolicyType
  QueueID::policy() const
  {
    return policy_;
  }

  inline
  size_t
  QueueID::index() const
  {
    return index_;
  }

  inline
  bool
  QueueID::isValid() const
  {
    return policy_ != enquing_policy::Max;
  }

  inline
  bool
  QueueID::operator< (QueueID const& other) const
  {
    return policy_ == other.policy_
      ? index_ < other.index_
      : policy_ < other.policy_;
  }

  inline
  bool
  QueueID::operator== (QueueID const& other) const
  {
    return policy_ == other.policy_ && index_ == other.index_;
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

#endif // EventFilter_StorageManager_QueueID_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
