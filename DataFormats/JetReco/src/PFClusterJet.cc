// $Id: PFClusterJet.cc,v 1.1 2010/12/28 16:11:35 srappocc Exp $

#include "DataFormats/JetReco/interface/PFClusterJet.h"



reco::PFClusterJet::PFClusterJet()
  : reco::Jet()
{
}


reco::PFClusterJet::PFClusterJet(const LorentzVector & fP4, const Point & fVertex) 
  : reco::Jet(fP4, fVertex)
{
}


reco::PFClusterJet::PFClusterJet(const LorentzVector & fP4,const Point & fVertex, const Jet::Constituents & fConstituents)
  : reco::Jet(fP4, fVertex, fConstituents)
{
}



reco::PFClusterJet * reco::PFClusterJet::clone() const {
  return new reco::PFClusterJet(*this);
}



reco::PFClusterRef reco::PFClusterJet::pfCluster(size_t i) const {
  Constituent dau = daughterPtr (i);
  // check the daughter to be ok
  if ( dau.isNonnull() && dau.isAvailable() ) {
    // convert to concrete candidate type
    const RecoPFClusterRefCandidate* pfClusterCand = dynamic_cast <const RecoPFClusterRefCandidate*> (dau.get());
    // check the candidate is of the right type
    if (pfClusterCand) {
      return pfClusterCand->pfCluster();
    } else {
     throw cms::Exception("Invalid Constituent") << "PFClusterJet constituent is not of RecoPFClusterRefCandidate type";
    }
  // otherwise return empty ptr
  } else {
    return reco::PFClusterRef();
  }
}



bool reco::PFClusterJet::overlap(const Candidate & dummy) const {
  return false;
}


std::string reco::PFClusterJet::print() const {
  std::ostringstream out;
  out << Jet::print() << std::endl;
  out << "    Constituents: " << std::endl;
  for ( size_t i = 0; i < numberOfDaughters(); ++i ) {
    out <<  *(pfCluster(i)) << std::endl;
  } 
  return out.str();
}
