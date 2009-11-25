// $Id$

#include "DataFormats/JetReco/interface/TrackJet.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"


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



edm::Ptr<reco::Track> reco::TrackJet::track(size_type i) const {
   Constituent dau = daughterPtr (i);

   if ( dau.isNonnull() && dau.isAvailable() ) {

   const RecoChargedRefCandidate* trkCand = dynamic_cast <const RecoChargedRefCandidate*> (dau.get());

    if (trkCand) {
//      return towerCandidate;
// 086     Ptr(ProductID const& productID, T const* item, key_type item_key) :
      return edm::Ptr<reco::Track> ( trkCand->track().id(), trkCand->track().get(), 
				     trkCand->track().key() );
    }
    else {
      throw cms::Exception("Invalid Constituent") << "TrackJet constituent is not of RecoChargedRefCandidatee type";
    }

   }

   else {
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
  for ( reco::Candidate::const_iterator idaBegin = this->begin(),
	  idaEnd = this->end(), ida = idaBegin;
	ida != idaEnd; ++ida ) {
    charge += ida->charge();
  }
  this->setCharge(charge);
}


bool reco::TrackJet::overlap(const Candidate & dummy) const {
  return false;
}


std::string reco::TrackJet::print() const {
  std::ostringstream out;
  out << Jet::print() // generic jet info
      << "    TrackJet specific: Printing not implemented yet" << std::endl;
  return out.str();
}
