// Jet.cc
// Fedor Ratnikov, UMd
// $Id: Jet.cc,v 1.3 2006/12/11 12:21:39 fedor Exp $

#include <sstream>
#include "PhysicsTools/Utilities/interface/DeltaR.h"

//Own header file
#include "DataFormats/JetReco/interface/Jet.h"

using namespace reco;

namespace {
  std::vector<const Candidate*> constituents (const reco::Jet& fJet) {
    std::vector<const Candidate*> result;
    int i = fJet.numberOfDaughters();
    if (i > 0) {
      CandidateRef ref = fJet.daughterRef (0);
      const CandidateCollection* container = ref.product();
      while (--i >= 0) {
	result.push_back (&((*container)[i]));
      }
    }
    return result;
  }
}

Jet::Jet (const LorentzVector& fP4, 
	  const Point& fVertex, 
	  const std::vector<reco::CandidateRef>& fConstituents)
  :  CompositeRefCandidate (0, fP4, fVertex)
{
  for (unsigned i = 0; i < fConstituents.size (); i++) addDaughter (fConstituents [i]);
}  


    /// eta-phi statistics
Jet::EtaPhiMoments Jet::etaPhiStatistics () const {
  std::vector<const Candidate*> towers = constituents (*this);
  double sumw = 0;
  double sumEta = 0;
  double sumEta2 = 0;
  double sumPhi = 0;
  double sumPhi2 = 0;
  double sumEtaPhi = 0;
  int i = towers.size();
  while (--i >= 0) {
    double eta = towers[i]->eta();
    double phi = towers[i]->eta();
    double weight = towers[i]->energy();
    sumw += weight;
    sumEta += eta * weight;
    sumEta2 += eta * eta * weight;
    sumPhi += phi * weight;
    sumPhi2 += phi * phi * weight;
    sumEtaPhi += eta * phi * weight;
  }
  Jet::EtaPhiMoments result;
  if (sumw > 0) {
    result.etaMean = sumEta / sumw;
    result.phiMean = sumPhi / sumw;
    result.etaEtaMoment = (sumEta2 - sumEta * sumEta / sumw) / sumw;
    result.phiPhiMoment = (sumPhi2 - sumPhi * sumPhi / sumw) / sumw;
    result.etaPhiMoment = (sumEtaPhi - sumEta * sumPhi / sumw) / sumw;
  }
  else {
    result.etaMean = 0;
    result.phiMean = 0;
    result.etaEtaMoment = 0;
    result.phiPhiMoment = 0;
    result.etaPhiMoment = 0;
  }
  return result;
}

/// eta-eta second moment
double Jet::etaetaMoment () const {
  std::vector<const Candidate*> towers = constituents (*this);
  double sumw = 0;
  double sum = 0;
  double sum2 = 0;
  int i = towers.size();
  while (--i >= 0) {
    double value = towers[i]->eta();
    double weight = towers[i]->energy();
    sumw += weight;
    sum += value * weight;
    sum2 += value * value * weight;
  }
  return sumw > 0 ? (sum2 - sum*sum/sumw ) / sumw : 0;
}

/// phi-phi second moment
double Jet::phiphiMoment () const {
  double sumw = 0;
  double sum = 0;
  double sum2 = 0;
  int i = numberOfDaughters();
  while (--i >= 0) {
    double value = daughter(i)->phi();
    double weight = daughter(i)->energy();
    sumw += weight;
    sum += value * weight;
    sum2 += value * value * weight;
  }
  return sumw > 0 ? (sum2 - sum*sum/sumw ) / sumw : 0;
}

/// eta-phi second moment
double Jet::etaphiMoment () const {
  Constituents towers = getJetConstituents ();
  double sumw = 0;
  double sumA = 0;
  double sumB = 0;
  double sumAB = 0;
  int i = towers.size();
  while (--i >= 0) {
    double valueA = towers[i]->eta();
    double valueB = towers[i]->phi();
    double weight = towers[i]->energy();
    sumw += weight;
    sumA += valueA * weight;
    sumB += valueB * weight;
    sumAB += valueA * valueB * weight;
  }
  return sumw > 0 ? (sumAB - sumA*sumB/sumw ) / sumw : 0;
}

  /// energy in annulus between rmin and rmax around jet direction
double Jet::energyInAnnulus (double fRmin, double fRmax) const {
  double result = 0;
  Constituents towers = getJetConstituents ();
  int i = towers.size ();
  while (--i >= 0) {
    double r = deltaR (*this, *(towers[i]));
    if (r >= fRmin && r < fRmax) result += towers[i]->energy ();
  }
  return result;
}

Jet::Constituents Jet::getJetConstituents () const {
  Jet::Constituents result;
  for (unsigned i = 0; i < CompositeRefCandidate::numberOfDaughters(); i++) {
    result.push_back (CompositeRefCandidate::daughterRef (i));
  }
  return result;
}

std::string Jet::print () const {
  std::ostringstream out;
  out << "Jet p/px/py/pz/pt: " << p() << '/' << px () << '/' << py() << '/' << pz() << '/' << pt() << std::endl
      << "    eta/phi: " << eta () << '/' << phi () << std::endl
       << "    # of constituents: " << nConstituents () << std::endl;
  out << "    Constituents:" << std::endl;
  Candidate::const_iterator daugh = begin ();
  int index = 0;
  for (; daugh != end (); daugh++, index++) {
    const Candidate* constituent = &*daugh; // deref
    if (constituent) {
      out << "      #" << index << " p/pt/eta/phi: " 
	  << constituent->p() << '/' << constituent->pt() << '/' << constituent->eta() << '/' << constituent->phi() << std::endl; 
    }
    else {
      out << "      #" << index << " constituent is not available in the event"  << std::endl;
    }
  }
  return out.str ();
}

