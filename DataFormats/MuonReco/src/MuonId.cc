#include "DataFormats/MuonReco/interface/MuonId.h"
using namespace reco;

MuonId::MuonId(  Charge q, const LorentzVector & p4, const Point & vtx ) : 
   Muon( q, p4, vtx ) 
{}
