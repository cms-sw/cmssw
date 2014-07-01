#include "DataFormats/BTauReco/interface/IPTagInfo.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/Candidate.h"
namespace reco {
template <> const reco::Track * reco::IPTagInfo<reco::TrackRefVector,reco::JTATagInfo>::selectedTrack(size_t i) const {return &(*m_selected[i]);}
template <> const reco::Track * reco::IPTagInfo<std::vector<reco::CandidatePtr>,reco::BaseTagInfo>::selectedTrack(size_t i) const {return (*m_selected[i]).bestTrack();}
}


