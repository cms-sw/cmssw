#ifndef Common_HLTenums_h
#define Common_HLTenums_h

/** \brief HLT enums
 *
 *  Definition of common HLT enums
 *
 *  $Date: 2006/04/11 10:10:10 $
 *  $Revision: 1.0 $
 *
 *  \author Martin Grunewald
 *
 */

namespace edm
{
  namespace hlt
    {

      // status of a trigger path
      enum HLTState {Ready=0,     // not [yet] run
		     Pass =1,     // accept
		     Fail =2,     // reject
		     Exception=3  // error
      };

      // predefined scalar physics observables
      enum HLTScalar {MET =0,     // total MET
		      METx=1,     // MET in x
		      METy=2,     // MET in y
		      METz=3,     // MET in z
		      ETOT=4,     // total energy
		      HT=5, ST=6 /* ... */ 
      };
    }
}

#endif // Common_HLTenums_h
