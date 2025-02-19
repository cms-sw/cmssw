// $Id: EnquingPolicyTag.h,v 1.4 2011/03/07 15:31:31 mommsen Exp $
/// @file: EnquingPolicyTag.h 

#ifndef EventFilter_StorageManager_EnquingPolicy_t
#define EventFilter_StorageManager_EnquingPolicy_t

#include <iostream>

namespace stor
{

  /**
     This enumeration is used to denote which queuing discipline is
     used for enquing items when the queue in question is full.

     $Author: mommsen $
     $Revision: 1.4 $
     $Date: 2011/03/07 15:31:31 $
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

#endif // EventFilter_StorageManager_EnquingPolicy_t

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -


