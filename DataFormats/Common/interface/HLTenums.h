#ifndef Common_HLTenums_h
#define Common_HLTenums_h

/** \brief HLT enums
 *
 *  Definition of common HLT enums
 *
 *  $Date: 2006/04/19 20:12:04 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

namespace edm
{
  namespace hlt
    {

      /// status of a trigger path
      enum HLTState {Ready=0,     ///< not [yet] run
		     Pass =1,     ///< accept
		     Fail =2,     ///< reject
		     Exception=3  ///< error
      };

    }
}

#endif // Common_HLTenums_h
