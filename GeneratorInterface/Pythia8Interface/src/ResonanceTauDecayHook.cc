#include "Pythia8/Pythia.h"
#include "GeneratorInterface/Pythia8Interface/interface/ResonanceTauDecayHook.h"

using namespace Pythia8;

bool ResonanceTauDecayHook::initAfterBeams() {
   filter_  = settingsPtr->flag("ResonanceTauDecayHook:use");
   if( filter_ ) decayer.init( infoPtr, settingsPtr, particleDataPtr, rndmPtr, (Couplings *) coupSMPtr );
   return true;
}

// Access the event after resonance decays.
bool ResonanceTauDecayHook::checkResonanceTauDecays(Event& process) {
   if( !filter_ ) return false;
   int procSize = process.size();
   for (int i = 0; i < procSize; ++i) {
      int iProc = process[i].idAbs();
      if ((iProc == 23 || iProc == 24 || iProc == 25) && process[i].status() == -22) {
        int iDec;
        // Find the tau
        int i1  = process[i].daughter1();
        int i2  = process[i].daughter2();
        if( process[i1].idAbs() == 15 ) {
           iDec = i1;
        } else if ( process[i2].idAbs() == 15 ) {
           iDec = i2;
        } else {
           return false;
        }
        // Send the tau off to be decayed:  if a pair, Pythia finds the sister
        decayer.decay(iDec, process);

      // End of loop over W/Z/H's. Do not veto any events.
      }
    }
   return false;    
}

