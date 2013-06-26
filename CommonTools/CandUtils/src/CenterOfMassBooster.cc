#include "CommonTools/CandUtils/interface/CenterOfMassBooster.h"
#include "DataFormats/Candidate/interface/Candidate.h"

CenterOfMassBooster::CenterOfMassBooster( const reco::Candidate & c ) : 
  booster( c.boostToCM() ) { 
}
