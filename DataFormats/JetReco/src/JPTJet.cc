// JPTJet.cc
// Fedor Ratnikov UMd
#include <sstream>

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"

//Own header file
#include "DataFormats/JetReco/interface/JPTJet.h"

using namespace reco;

JPTJet::JPTJet(const LorentzVector& fP4,
               const Point& fVertex,
               const Specific& fSpecific,
               const Jet::Constituents& fConstituents)
    : Jet(fP4, fVertex), mspecific(fSpecific) {}

JPTJet::JPTJet(const LorentzVector& fP4, const Specific& fSpecific, const Jet::Constituents& fConstituents)
    : Jet(fP4, Point(0, 0, 0)), mspecific(fSpecific) {}

JPTJet* JPTJet::clone() const { return new JPTJet(*this); }

bool JPTJet::overlap(const Candidate&) const { return false; }

void JPTJet::printJet() const {
  std::cout << " Raw Calo jet " << getCaloJetRef()->et() << " " << getCaloJetRef()->eta() << " "
            << getCaloJetRef()->phi() << "    JPTJet specific:" << std::endl
            << "      charged multiplicity: " << chargedMultiplicity() << std::endl;
  std::cout << "      JPTCandidate constituents:" << std::endl;
  std::cout << " Number of pions: " << getPionsInVertexInCalo().size() + getPionsInVertexOutCalo().size() << std::endl;
  std::cout << " Number of muons: " << getMuonsInVertexInCalo().size() + getMuonsInVertexOutCalo().size() << std::endl;
  std::cout << " Number of Electrons: " << getElecsInVertexInCalo().size() + getElecsInVertexOutCalo().size()
            << std::endl;
}

std::string JPTJet::print() const {
  std::ostringstream out;
  out << Jet::print()  // generic jet info
      << "    JPTJet specific:" << std::endl
      << "      charged: " << chargedMultiplicity() << std::endl;
  out << "      JPTCandidate constituents:" << std::endl;

  out << " Number of pions: " << getPionsInVertexInCalo().size() + getPionsInVertexOutCalo().size() << std::endl;
  out << " Number of muons: " << getMuonsInVertexInCalo().size() + getMuonsInVertexOutCalo().size() << std::endl;
  out << " Number of Electrons: " << getElecsInVertexInCalo().size() + getElecsInVertexOutCalo().size() << std::endl;

  return out.str();
}
