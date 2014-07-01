#include "GeneratorInterface/PartonShowerVeto/interface/SetNumberOfPartonsDynamically.h"

using namespace Pythia8;

bool SetNumberOfPartonsDynamically::doVetoProcessLevel(Event& process) {
  
  int nPartons = 0;
  // Do not include resonance decay products in the counting.
  omitResonanceDecays(process);
  // Get the maximal quark flavour counted as "additional" parton.
  int nQuarksMerge = settingsPtr->mode("Merging:nQuarksMerge");
  // Loop through event and count.
  for(int i=0; i < int(workEvent.size()); ++i)
    if ( workEvent[i].isFinal()
      && workEvent[i].colType()!= 0
      && ( workEvent[i].id() == 21 || workEvent[i].idAbs() <= nQuarksMerge))
      nPartons++;
  // Set number of requested partons.
  settingsPtr->mode("Merging:nRequested", nPartons);
  // For UMEPS, also remove zero-parton contributions from prospective
  // subtractive events.
  bool doSubt = settingsPtr->flag("Merging:doUMEPSSubt")
            || settingsPtr->flag("Merging:doUNLOPSSubt");
  if (doSubt && nPartons == 0) return true;
  // Done
  return false;
  
}