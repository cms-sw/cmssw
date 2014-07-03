#include "DataFormats/BTauReco/interface/IPTagInfo.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/Candidate.h"
namespace reco {
namespace btag {
 template <> const reco::Track * toTrack(const reco::TrackRef & t) {return &(*t);}
 template <> const reco::Track * toTrack(const reco::CandidatePtr & c) {return (*c).bestTrack();}

}

//template <> const reco::Track * reco::IPTagInfo<reco::TrackRefVector,reco::JTATagInfo>::selectedTrack(size_t i) const {return &(*m_selected[i]);}
//template <> const reco::Track * reco::IPTagInfo<std::vector<reco::CandidatePtr>,reco::JetTagInfo>::selectedTrack(size_t i) const {return (*m_selected[i]).bestTrack();}
}


