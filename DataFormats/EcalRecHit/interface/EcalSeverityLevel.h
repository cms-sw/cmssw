/**
   \file EcalSeverityLevel
   

   \author Stefano Argiro
   \version $Id: EcalSeverityLevel.h,v 1.1 2011/04/11 14:51:51 argiro Exp $
   \date 11 Apr 2011
*/

#ifndef __EcalSeverityLevel_h_
#define __EcalSeverityLevel_h_

/** Define how good a rechit is to be used for reco.
   
 */

namespace EcalSeverityLevel {

  enum SeverityLevel {
    kGood=0,             // good channel 
    kProblematic,        // problematic (e.g. noisy)
    kRecovered,          // recovered (e.g. an originally dead or saturated)
    kTime,               // the channel is out of time (e.g. spike)
    kWeird,              // weird (e.g. spike)
    kBad                 // bad, not suitable to be used for reconstruction
  };
      
  
}

#endif // __EcalSeverityLevel_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "cd .. ; scram b"
// End:
