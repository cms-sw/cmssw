// GenJet.cc
// Fedor Ratnikov, UMd
// $Id: GenJet.cc,v 1.12 2008/05/26 11:22:12 arizzi Exp $

#include <sstream>

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

//Own header file
#include "DataFormats/JetReco/interface/GenJet.h"

using namespace reco;

GenJet::GenJet (const LorentzVector& fP4, const Point& fVertex, 
		const Specific& fSpecific)
  : Jet (fP4, fVertex),
    m_specific (fSpecific)
{}

GenJet::GenJet (const LorentzVector& fP4, const Point& fVertex, 
		const Specific& fSpecific, 
		const Jet::Constituents& fConstituents)
  : Jet (fP4, fVertex, fConstituents),
    m_specific (fSpecific)
{}

GenJet::GenJet (const LorentzVector& fP4, 
		const Specific& fSpecific, 
		const Jet::Constituents& fConstituents)
  : Jet (fP4, Point(0,0,0), fConstituents),
    m_specific (fSpecific)
{}

/// Detector Eta (use reference Z and jet kinematics only)
float GenJet::detectorEta (float fZVertex) const {
  return Jet::detectorEta (fZVertex, eta());
}

const GenParticle* GenJet::genParticle (const Candidate* fConstituent) {
  const Candidate* base = fConstituent;
  if (fConstituent->hasMasterClone ()) base = fConstituent->masterClone().get ();
  const GenParticle* result = dynamic_cast<const GenParticle*> (base);
  if (!result) throw cms::Exception("Invalid Constituent") << "GenJet constituent is not of the type GenParticle";
  return result;
}

const GenParticle* GenJet::getGenConstituent (unsigned fIndex) const {
  // no direct access, have to iterate for now
  int index (fIndex);
  Candidate::const_iterator daugh = begin ();
  for (; --index >= 0 && daugh != end (); daugh++) {}
  if (daugh != end ()) { // in range
    const Candidate* constituent = &*daugh; // deref
    return genParticle (constituent);
  }
  return 0;
}

std::vector <const GenParticle*> GenJet::getGenConstituents () const {
  std::vector <const GenParticle*> result;
  for (unsigned i = 0;  i <  numberOfDaughters (); i++) result.push_back (getGenConstituent (i));
  return result;
}

GenJet* GenJet::clone () const {
  return new GenJet (*this);
}

bool GenJet::overlap( const Candidate & ) const {
  return false;
}

std::string GenJet::print () const {
  std::ostringstream out;
  out << Jet::print () // generic jet info
      << "    GenJet specific:" << std::endl
      << "      em/had/invisible/aux  energies: " 
      << emEnergy() << '/' << hadEnergy()  << '/' << invisibleEnergy() << '/' << auxiliaryEnergy() << std::endl;
  out << "      MC particles:" << std::endl;
  std::vector <const GenParticle*> mcparts = getGenConstituents ();
  for (unsigned i = 0; i < mcparts.size (); i++) {
    const GenParticle* mcpart = mcparts[i];
    if (mcpart) {
      out << "      #" << i << "  PDG code:" << mcpart->pdgId() 
	  << ", p/pt/eta/phi: " << mcpart->p() << '/' << mcpart->pt() << '/' << mcpart->eta() << '/' << mcpart->phi() << std::endl;
    }
    else {
      out << "      #" << i << "  No information about constituent" << std::endl;
    }
  }
  return out.str ();
}
