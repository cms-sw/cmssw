// JPTJet.cc
// Fedor Ratnikov UMd
// $Id: JPTJet.cc,v 1.22 2009/04/16 20:04:20 srappocc Exp $
#include <sstream>

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"

//Own header file
#include "DataFormats/JetReco/interface/JPTJet.h"

using namespace reco;

JPTJet::JPTJet (const LorentzVector& fP4, const Point& fVertex, 
		const Specific& fSpecific, const Jet::Constituents& fConstituents 
		)
  : Jet (fP4, fVertex),
    m_specific (fSpecific)
{}

JPTJet::JPTJet (const LorentzVector& fP4, 
		const Specific& fSpecific, const Jet::Constituents& fConstituents 
		)
  : Jet (fP4, Point(0,0,0)),
    m_specific (fSpecific)
{}


JPTJet* JPTJet::clone () const {
  return new JPTJet (*this);
}

bool JPTJet::overlap( const Candidate & ) const {
  return false;
}


void JPTJet::printJet () const {
  std::cout <<  " Raw Calo jet " <<getCaloJetRef()->et()<<" "<<getCaloJetRef()->eta()<<" "<<getCaloJetRef()->phi()
      << "    JPTJet specific:" << std::endl
      << "      chargedhadrons energy: " << chargedHadronEnergy () << std::endl
      << "      charged multiplicity: " << chargedMultiplicity () << std::endl;
  std::cout << "      JPTCandidate constituents:" << std::endl;
  std::cout<< " Number of pions: "<< getPions_inVertexInCalo().size()+getPions_inVertexOutCalo().size()<<std::endl;
  std::cout<< " Number of muons: "<< getMuons_inVertexInCalo().size()+getMuons_inVertexOutCalo().size()<<std::endl;
  std::cout<< " Number of Electrons: "<< getElecs_inVertexInCalo().size()+getElecs_inVertexOutCalo().size()<<std::endl;
  
}

std::string JPTJet::print () const {
  std::ostringstream out;
  out << Jet::print () // generic jet info
      << "    JPTJet specific:" << std::endl
      << "      chargedhadrons energy: " << chargedHadronEnergy () << std::endl
      << "      charged: " << chargedMultiplicity () << std::endl;
  out << "      JPTCandidate constituents:" << std::endl;

  out<< " Number of pions: "<< getPions_inVertexInCalo().size()+getPions_inVertexOutCalo().size()<<std::endl;
  out<< " Number of muons: "<< getMuons_inVertexInCalo().size()+getMuons_inVertexOutCalo().size()<<std::endl;
  out<< " Number of Electrons: "<< getElecs_inVertexInCalo().size()+getElecs_inVertexOutCalo().size()<<std::endl;
  
  return out.str ();
}
