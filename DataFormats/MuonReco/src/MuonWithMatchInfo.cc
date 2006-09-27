#include "DataFormats/MuonReco/interface/MuonWithMatchInfo.h"
using namespace reco;

MuonWithMatchInfo::MuonWithMatchInfo(  Charge q, const LorentzVector & p4, const Point & vtx ) : 
   Muon( q, p4, vtx ) 
{}
