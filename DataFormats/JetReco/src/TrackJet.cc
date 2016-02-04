// $Id: TrackJet.cc,v 1.2 2009/12/10 15:13:43 lowette Exp $

#include "DataFormats/JetReco/interface/TrackJet.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"


reco::TrackJet::TrackJet()
  : reco::Jet()
{
}


reco::TrackJet::TrackJet(const LorentzVector & fP4, const Point & fVertex) 
  : reco::Jet(fP4, fVertex)
{
}


reco::TrackJet::TrackJet(const LorentzVector & fP4, const Point & fVertex, const Jet::Constituents & fConstituents)
  : reco::Jet(fP4, fVertex, fConstituents)
{
  this->resetCharge();
}



reco::TrackJet * reco::TrackJet::clone() const {
  return new reco::TrackJet(*this);
}



edm::Ptr<reco::Track> reco::TrackJet::track(size_t i) const {
  Constituent dau = daughterPtr (i);
  // check the daughter to be ok
  if ( dau.isNonnull() && dau.isAvailable() ) {
    // convert to concrete candidate type
    const RecoChargedRefCandidate* trkCand = dynamic_cast <const RecoChargedRefCandidate*> (dau.get());
    // check the candidate is of the right type
    if (trkCand) {
      // check the track link in the recochargedcandidate to be there
      if (trkCand->track().get()) {
        // ok, return pointer to the originating track
        return edm::Ptr<reco::Track> ( trkCand->track().id(), trkCand->track().get(), trkCand->track().key() );
      } else {
        throw cms::Exception("TrackRef unavailable") << "TrackJet consituent track not in the event.";
      }
    } else {
     throw cms::Exception("Invalid Constituent") << "TrackJet constituent is not of RecoChargedRefCandidate type";
    }
  // otherwise return empty ptr
  } else {
    return edm::Ptr<reco::Track>();
  }
}


std::vector<edm::Ptr<reco::Track> > reco::TrackJet::tracks() const {
  std::vector <edm::Ptr<reco::Track> > result;
  for (unsigned i = 0;  i <  numberOfDaughters (); i++) result.push_back (track (i));
  return result;
}


void reco::TrackJet::resetCharge() {
  reco::LeafCandidate::Charge charge = 0;
  for ( reco::Candidate::const_iterator ida = this->begin(); ida != this->end(); ++ida ) {
    charge += ida->charge();
  }
  this->setCharge(charge);
}


const reco::VertexRef reco::TrackJet::primaryVertex() const {
  return vtx_;
}


void reco::TrackJet::setPrimaryVertex(const reco::VertexRef & vtx) {
  vtx_ = vtx;
}


bool reco::TrackJet::overlap(const Candidate & dummy) const {
  return false;
}


std::string reco::TrackJet::print() const {
  std::ostringstream out;
  out << Jet::print() // generic jet info
      << "    TrackJet specific:" << std::endl;
  if (primaryVertex().get()) {
    out << "      Associated PV:"
        << " x=" << primaryVertex()->x()
        << " y=" << primaryVertex()->y()
        << " z=" << primaryVertex()->z() << std::endl;
  } else {
    out << "      Associated PV not available on the event" << std::endl;
  }
  std::vector<edm::Ptr<reco::Track> > thetracks = tracks();
  for (unsigned i = 0; i < thetracks.size (); i++) {
    if (thetracks[i].get ()) {
      out << "      #" << i
          << " px=" << thetracks[i]->px()
          << " py=" << thetracks[i]->py()
          << " pz=" << thetracks[i]->pz()
          << " eta=" << thetracks[i]->eta()
          << " phi=" << thetracks[i]->phi() << std::endl;
    }
    else {
      out << "      #" << i << " track is not available in the event"  << std::endl;
    }
  }
  return out.str();
}
