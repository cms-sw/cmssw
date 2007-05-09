// Jet.cc
// Fedor Ratnikov, UMd
// $Id: Jet.cc,v 1.6 2007/05/08 21:36:52 fedor Exp $

#include <sstream>
#include "PhysicsTools/Utilities/interface/Math.h"

//Own header file
#include "DataFormats/JetReco/interface/Jet.h"

using namespace reco;

namespace {
}

Jet::Jet (const LorentzVector& fP4, 
	  const Point& fVertex, 
	  const std::vector<reco::CandidateRef>& fConstituents)
  :  CompositeRefCandidate (0, fP4, fVertex)
{
  for (unsigned i = 0; i < fConstituents.size (); i++) addDaughter (fConstituents [i]);
}  

/// return # of constituent carring fraction of energy. Assume ordered towers
int Jet::nCarrying (double fFraction) const {
  std::vector<const Candidate*> towers = getJetConstituentsQuick ();
  if (fFraction >= 1) return towers.size();
  double totalEt = 0;
  for (unsigned i = 0; i < towers.size(); ++i) totalEt += towers[i]->et();
  double fractionEnergy = totalEt * fFraction;
  unsigned result = 0;
  for (; result < towers.size(); ++result) {
    fractionEnergy -= towers[result]->energy();
    if (fractionEnergy <= 0) return result+1;
  }
  return 0;
}

/// eta-phi statistics
Jet::EtaPhiMoments Jet::etaPhiStatistics () const {
  Constituents towers = getJetConstituents ();
  double phiRef = phi();
  double sumw = 0;
  double sumEta = 0;
  double sumEta2 = 0;
  double sumPhi = 0;
  double sumPhi2 = 0;
  double sumEtaPhi = 0;
  int i = towers.size();
  while (--i >= 0) {
    double eta = towers[i]->eta();
    double phi = deltaPhi (towers[i]->phi(), phiRef);
    double weight = towers[i]->et();
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
    result.phiMean = deltaPhi (phiRef + sumPhi, 0.);
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
  Constituents towers = getJetConstituents ();
  double sumw = 0;
  double sum = 0;
  double sum2 = 0;
  int i = towers.size();
  while (--i >= 0) {
    double value = towers[i]->eta();
    double weight = towers[i]->et();
    sumw += weight;
    sum += value * weight;
    sum2 += value * value * weight;
  }
  return sumw > 0 ? (sum2 - sum*sum/sumw ) / sumw : 0;
}

/// phi-phi second moment
double Jet::phiphiMoment () const {
  double phiRef = phi();
  Constituents towers = getJetConstituents ();
  double sumw = 0;
  double sum = 0;
  double sum2 = 0;
  int i = towers.size();
  while (--i >= 0) {
    double value = deltaPhi (towers[i]->phi(), phiRef);
    double weight = towers[i]->et();
    sumw += weight;
    sum += value * weight;
    sum2 += value * value * weight;
  }
  return sumw > 0 ? (sum2 - sum*sum/sumw ) / sumw : 0;
}

/// eta-phi second moment
double Jet::etaphiMoment () const {
  double phiRef = phi();
  Constituents towers = getJetConstituents ();
  double sumw = 0;
  double sumA = 0;
  double sumB = 0;
  double sumAB = 0;
  int i = towers.size();
  while (--i >= 0) {
    double valueA = towers[i]->eta();
    double valueB = deltaPhi (towers[i]->phi(), phiRef);
    double weight = towers[i]->et();
    sumw += weight;
    sumA += valueA * weight;
    sumB += valueB * weight;
    sumAB += valueA * valueB * weight;
  }
  return sumw > 0 ? (sumAB - sumA*sumB/sumw ) / sumw : 0;
}

/// et in annulus between rmin and rmax around jet direction
double Jet::etInAnnulus (double fRmin, double fRmax) const {
  double result = 0;
  Constituents towers = getJetConstituents ();
  int i = towers.size ();
  while (--i >= 0) {
    double r = deltaR (*this, *(towers[i]));
    if (r >= fRmin && r < fRmax) result += towers[i]->et ();
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

std::vector<const Candidate*> Jet::getJetConstituentsQuick () const {
  std::vector<const Candidate*> result;
  int i = numberOfDaughters();
  if (i > 0) {
    CandidateRef ref = daughterRef (0);
    const CandidateCollection* container = ref.product();
    while (--i >= 0) {
      result.push_back (&((*container)[i]));
    }
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

