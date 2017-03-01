#include "DataFormats/PatCandidates/interface/libminifloat.h"
#include "DataFormats/Candidate/interface/Candidate.h"
int16_t convertPackedEtaToPackedY(int16_t packedPt_, int16_t packedEta_,int16_t packedM_)
{
 reco::Candidate::PolarLorentzVector p4(MiniFloatConverter::float16to32(packedPt_),
                         int16_t(packedEta_)*6.0f/std::numeric_limits<int16_t>::max(),
                         0,
                         MiniFloatConverter::float16to32(packedM_));

 return int16_t(p4.Rapidity()/6.0f*std::numeric_limits<int16_t>::max());
}

