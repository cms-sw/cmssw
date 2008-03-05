#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrackMuon.h"

#include <iomanip>

using namespace reco;
using namespace std;


PFBlockElementTrackMuon::PFBlockElementTrackMuon( const reco::MuonRef& muonref,
						  const PFRecTrackRef& ref) :
  PFBlockElementTrack( ref, MUON ),
  muonRef_( muonref ) {}


void PFBlockElementTrackMuon::Dump(ostream& out, 
				   const char* tab ) const {
  
  PFBlockElementTrack::Dump(out, tab);
  
  // if you want to dump some muon-related quantities 
  // you should add it here (marcella bona 2008/02/20)

}
