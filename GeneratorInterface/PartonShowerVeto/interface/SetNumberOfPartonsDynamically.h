// This program is written by Stefan Prestel.
// It illustrates how to do run PYTHIA with LHEF input, allowing a
// sample-by-sample generation of
// a) Non-matched/non-merged events
// b) MLM jet-matched events (kT-MLM, shower-kT, FxFx)
// c) CKKW-L and UMEPS-merged events
// d) UNLOPS (and NL3) NLO merged events
// see the respective sections in the online manual for details.

#ifndef Pythia8_SetNumberOfPartonsDynamically_H
#define Pythia8_SetNumberOfPartonsDynamically_H

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"
#include <unistd.h>

#include "HepMC/GenEvent.h"
#include "HepMC/IO_GenEvent.h"
// Following line to be used with HepMC 2.04 onwards.
#include "HepMC/Units.h"

// Include UserHooks for Jet Matching.
#include "GeneratorInterface/PartonShowerVeto/interface/CombineMatchingInput.h"

//==========================================================================

// Use userhooks to set the number of requested partons dynamically, as
// needed when running CKKW-L or UMEPS on a single input file that contains
// all parton multiplicities.

class SetNumberOfPartonsDynamically : public Pythia8::UserHooks {

public:

 // Constructor and destructor.
 SetNumberOfPartonsDynamically() {}
~SetNumberOfPartonsDynamically() {}

 // Allow to set the number of partons.
 bool canVetoProcessLevel() { return true; }
 // Set the number of partons.
 bool doVetoProcessLevel(Pythia8::Event& process);

private:

};
#endif
