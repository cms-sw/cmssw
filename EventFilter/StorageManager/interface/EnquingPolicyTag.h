// $Id$

#ifndef EventFilter_StorageManager_EnquingPolicy_t
#define EventFilter_StorageManager_EnquingPolicy_t

#include <iostream>

namespace stor
{

  /**
     This enumeration is used to denote which queuing discipline is
     used for enquing items when the queue in question is full.

     $Author$
     $Revision$
     $Date$
   */

  namespace enquing_policy
    {
      enum PolicyTag
	{
	  DiscardNew,
	  DiscardOld,
	  FailIfFull,
	  Max
	};

      std::ostream& operator << ( std::ostream& os,
                                  const enquing_policy::PolicyTag& ptag );

  } // namespace enquing_policy

} // namespace stor

#endif

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -


