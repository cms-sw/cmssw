#include "DataFormats/MuonReco/interface/Muon.h"
using namespace reco;

Muon::Muon(  Charge q, const LorentzVector & p4, const Point & vtx ) : 
      RecoCandidate( q, p4, vtx ) { 
}


