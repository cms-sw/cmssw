// $Id: EnquingPolicyTag.h,v 1.2 2009/06/10 08:15:22 dshpakov Exp $
/// @file: EnquingPolicyTag.h 

#ifndef EventFilter_StorageManager_EnquingPolicy_t
#define EventFilter_StorageManager_EnquingPolicy_t

#include <iostream>

namespace stor
{

  /**
     This enumeration is used to denote which queuing discipline is
     used for enquing items when the queue in question is full.

     $Author: dshpakov $
     $Revision: 1.2 $
     $Date: 2009/06/10 08:15:22 $
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


