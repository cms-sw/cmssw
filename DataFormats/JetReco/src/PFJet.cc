// PFJet.cc
// Fedor Ratnikov UMd
// $Id: PFJet.cc,v 1.17 2010/03/10 21:52:18 pandolf Exp $
#include <sstream>
#include <typeinfo>

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

//Own header file
#include "DataFormats/JetReco/interface/PFJet.h"

using namespace reco;

PFJet::PFJet (const LorentzVector& fP4, const Point& fVertex, 
		  const Specific& fSpecific, 
		  const Jet::Constituents& fConstituents)
  : Jet (fP4, fVertex, fConstituents),
    m_specific (fSpecific)
{}

PFJet::PFJet (const LorentzVector& fP4, const Point& fVertex, 
	      const Specific& fSpecific)
  : Jet (fP4, fVertex),
    m_specific (fSpecific)
{}

PFJet::PFJet (const LorentzVector& fP4, 
		  const Specific& fSpecific, 
		  const Jet::Constituents& fConstituents)
  : Jet (fP4, Point(0,0,0), fConstituents),
    m_specific (fSpecific)
{}

reco::PFCandidatePtr PFJet::getPFConstituent (unsigned fIndex) const {

  Constituent dau = daughterPtr (fIndex);
  if ( dau.isNonnull() && dau.isAvailable() ) {
    const PFCandidate* pfCandidate = dynamic_cast <const PFCandidate*> (dau.get());
    if (pfCandidate) {
      return edm::Ptr<PFCandidate> (dau.id(), pfCandidate, dau.key() );
    }
    else {
      throw cms::Exception("Invalid Constituent") << "PFJet constituent is not of PFCandidate type";
    }
   }
   else {
     return PFCandidatePtr();
   }
}

std::vector <reco::PFCandidatePtr> PFJet::getPFConstituents () const {
  std::vector <PFCandidatePtr> result;
  for (unsigned i = 0;  i <  numberOfDaughters (); i++) result.push_back (getPFConstituent(i));
  return result;
}


reco::TrackRefVector PFJet::getTrackRefs() const {
  // result will contain chargedMultiplicity() elements
  reco::TrackRefVector result;
  result.reserve( chargedMultiplicity() );
  for (unsigned i = 0;  i <  numberOfDaughters (); i++) {
    const reco::PFCandidatePtr pfcand = getPFConstituent (i);
    reco::TrackRef trackref = pfcand->trackRef();
    if( trackref.isNonnull() ) {
      result.push_back( trackref );
    }
  }

  return result;
}


PFJet* PFJet::clone () const {
  return new PFJet (*this);
}

bool PFJet::overlap( const Candidate & ) const {
  return false;
}

std::string PFJet::print () const {
  std::ostringstream out;
  out << Jet::print () // generic jet info
      << "    PFJet specific:" << std::endl
      << "      charged hadron energy/multiplicity: " << chargedHadronEnergy () << '/' << chargedHadronMultiplicity () << std::endl
      << "      neutral hadron energy/multiplicity: " << neutralHadronEnergy () << '/' << neutralHadronMultiplicity () << std::endl
      << "      photon energy/multiplicity: " << photonEnergy () << '/' << photonMultiplicity () << std::endl
      << "      electron energy/multiplicity: " << electronEnergy () << '/' << electronMultiplicity () << std::endl
      << "      muon energy/multiplicity: " << muonEnergy () << '/' << muonMultiplicity () << std::endl
      << "      HF Hadron energy/multiplicity: " << HFHadronEnergy () << '/' << HFHadronMultiplicity () << std::endl
      << "      HF EM particle energy/multiplicity: " << HFEMEnergy () << '/' << HFEMMultiplicity () << std::endl
      << "      charged/neutral hadrons energy: " << chargedHadronEnergy () << '/' << neutralHadronEnergy () << std::endl
      << "      charged/neutral em energy: " << chargedEmEnergy () << '/' << neutralEmEnergy () << std::endl
      << "      charged muon energy: " << chargedMuEnergy () << '/' << std::endl
      << "      charged/neutral multiplicity: " << chargedMultiplicity () << '/' << neutralMultiplicity () << std::endl;
  out << "      PFCandidate constituents:" << std::endl;
  std::vector <PFCandidatePtr> constituents = getPFConstituents ();
  for (unsigned i = 0; i < constituents.size (); ++i) {
    if (constituents[i].get()) {
      out << "      #" << i << " " << *(constituents[i]) << std::endl;
    }
    else {
      out << "      #" << i << " PFCandidate is not available in the event"  << std::endl;
    }
  }
  return out.str ();
}

std::ostream& reco::operator<<(std::ostream& out, const reco::PFJet& jet) {

  if(out) {
    out<<"PFJet "
       <<"(pt, eta, phi) = "<<jet.pt()<<","<<jet.eta()<<","<<jet.phi()
       <<"  (Rch,Rnh,Rgamma,Re,Rmu,RHFHad,RHFEM) = "
       <<jet.chargedHadronEnergyFraction()<<","
       <<jet.neutralHadronEnergyFraction()<<","
       <<jet.photonEnergyFraction()<<","
       <<jet.electronEnergyFraction()<<","
       <<jet.muonEnergyFraction()<<","
       <<jet.HFHadronEnergyFraction()<<","
       <<jet.HFEMEnergyFraction();
  }
  return out;
}
