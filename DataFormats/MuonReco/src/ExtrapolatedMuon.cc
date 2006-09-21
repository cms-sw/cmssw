#include "DataFormats/MuonReco/interface/ExtrapolatedMuon.h"
using namespace reco;

ExtrapolatedMuon::ExtrapolatedMuon(  Charge q, const LorentzVector & p4, const Point & vtx ) : 
   Muon( q, p4, vtx ) 
{}
