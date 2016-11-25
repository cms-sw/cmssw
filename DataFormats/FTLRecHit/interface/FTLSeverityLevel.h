/**
   \file FTLSeverityLevel
   

*/

#ifndef __DataFormats_FTLSeverityLevel_h_
#define __DataFormats_FTLSeverityLevel_h_

/** Define how good a rechit is to be used for reco.
   
 */

namespace FTLSeverityLevel {

  enum SeverityLevel {
    kGood=0,             // good channel 
    kProblematic,        // problematic (e.g. noisy)
    kRecovered,          // recovered (e.g. an originally dead or saturated)
    kTime,               // the channel is out of time (e.g. spike)
    kWeird,              // weird (e.g. spike)
    kBad                 // bad, not suitable to be used for reconstruction
  };
      
  
}

#endif // __FTLSeverityLevel_h_

