// Jet.cc
// Fedor Ratnikov, UMd
// $Id: Jet.cc,v 1.20 2008/02/14 18:27:20 fedor Exp $

#include <sstream>
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

//Own header file
#include "DataFormats/JetReco/interface/Jet.h"

using namespace reco;

namespace {
  // approximate simple CALO geometry
  class CaloPoint {
  public:
    CaloPoint (double fZ, double fEta) {
      const double depth = 0.1; // one for all relative depth of the reference point between ECAL begin and HCAL end
      const double R_BARREL = (1.-depth)*143.+depth*407.;
      const double Z_ENDCAP = (1.-depth)*320.+depth*568.; // 1/2(EEz+HEz)
      const double R_FORWARD = Z_ENDCAP / sqrt (cosh(3.)*cosh(3.) -1.); // eta=3
      const double Z_FORWARD = 1100.+depth*165.;
      const double ETA_MAX = 5.2;
      const double Z_BIG = 1.e5;
      
      if (fZ > Z_ENDCAP) fZ = Z_ENDCAP-1.;
      if (fZ < -Z_ENDCAP) fZ = -Z_ENDCAP+1; // sanity check
      
      double tanThetaAbs = sqrt (cosh(fEta)*cosh(fEta) - 1.);
      double tanTheta = fEta >= 0 ? tanThetaAbs : -tanThetaAbs;
      
      double rEndcap = tanTheta == 0 ? 1.e10 : 
	fEta > 0 ? (Z_ENDCAP - fZ) / tanTheta : (-Z_ENDCAP - fZ) / tanTheta;
      if (rEndcap > R_BARREL) { // barrel
	mR = R_BARREL;
	mZ = fZ + R_BARREL * tanTheta; 
      }
      else {
	double zRef = Z_BIG; // very forward;
	if (rEndcap > R_FORWARD) zRef = Z_ENDCAP; // endcap
	else if (fabs (fEta) < ETA_MAX) zRef = Z_FORWARD; // forward
	
	mZ = fEta > 0 ? zRef : -zRef;
	mR = fabs ((mZ - fZ) / tanTheta);
      }
    }

    double etaReference (double fZ) {
      Jet::Point p (r(), 0., z() - fZ);
      return p.eta();
    }

    double thetaReference (double fZ) {
      Jet::Point p (r(), 0., z() - fZ);
      return p.theta();
    }

    double z() const {return mZ;}
    double r() const {return mR;}

  private:
    double mZ;
    double mR;
  };
}

Jet::Jet (const LorentzVector& fP4, 
	  const Point& fVertex, 
	  const Constituents& fConstituents)
  :  CompositeRefBaseCandidate (0, fP4, fVertex),
     mJetArea (0),
     mPileupEnergy (0),
     mPassNumber (0)
{
  for (unsigned i = 0; i < fConstituents.size (); i++) {
    addDaughter (fConstituents [i]);
  }
}  

Jet::Jet (const LorentzVector& fP4, 
	  const Point& fVertex) 
  :  CompositeRefBaseCandidate (0, fP4, fVertex),
     mJetArea (0),
     mPileupEnergy (0),
     mPassNumber (0)
{}

/// eta-phi statistics
Jet::EtaPhiMoments Jet::etaPhiStatistics () const {
  std::vector<const Candidate*> towers = getJetConstituentsQuick ();
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
float Jet::etaetaMoment () const {
  std::vector<const Candidate*> towers = getJetConstituentsQuick ();
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
float Jet::phiphiMoment () const {
  double phiRef = phi();
  std::vector<const Candidate*> towers = getJetConstituentsQuick ();
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
float Jet::etaphiMoment () const {
  double phiRef = phi();
  std::vector<const Candidate*> towers = getJetConstituentsQuick ();
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
float Jet::etInAnnulus (float fRmin, float fRmax) const {
  float result = 0;
  std::vector<const Candidate*> towers = getJetConstituentsQuick ();
  int i = towers.size ();
  while (--i >= 0) {
    double r = deltaR (*this, *(towers[i]));
    if (r >= fRmin && r < fRmax) result += towers[i]->et ();
  }
  return result;
}

/// return # of constituent carring fraction of energy. Assume ordered towers
int Jet::nCarrying (float fFraction) const {
  std::vector<const Candidate*> towers = getJetConstituentsQuick ();
  if (fFraction >= 1) return towers.size();
  double totalEt = 0;
  for (unsigned i = 0; i < towers.size(); ++i) totalEt += towers[i]->et();
  double fractionEnergy = totalEt * fFraction;
  unsigned result = 0;
  for (; result < towers.size(); ++result) {
    fractionEnergy -= towers[result]->et();
    if (fractionEnergy <= 0) return result+1;
  }
  return 0;
}

    /// maximum distance from jet to constituent
float Jet::maxDistance () const {
  float result = 0;
  std::vector<const Candidate*> towers = getJetConstituentsQuick ();
  for (unsigned i = 0; i < towers.size(); ++i) {
    float dR = deltaR (*(towers[i]), *this);
    if (dR > result)  result = dR;
  }
  return result;
}

/// static function to convert detector eta to physics eta
float Jet::physicsEta (float fZVertex, float fDetectorEta) {
  CaloPoint refPoint (0., fDetectorEta);
  return refPoint.etaReference (fZVertex);
}

/// static function to convert physics eta to detector eta
float Jet::detectorEta (float fZVertex, float fPhysicsEta) {
  CaloPoint refPoint (fZVertex, fPhysicsEta);
  return refPoint.etaReference (0.);
}

Jet::Constituents Jet::getJetConstituents () const {
  Jet::Constituents result;
  for (unsigned i = 0; i < numberOfDaughters(); i++) {
    result.push_back (daughterRef (i));
  }
  return result;
}

std::vector<const Candidate*> Jet::getJetConstituentsQuick () const {
  std::vector<const Candidate*> result;
  int nDaughters = numberOfDaughters();
  for (int i = 0; i < nDaughters; ++i) { 
    result.push_back (daughter (i));
  }
  return result;
}

std::string Jet::print () const {
  std::ostringstream out;
  out << "Jet p/px/py/pz/pt: " << p() << '/' << px () << '/' << py() << '/' << pz() << '/' << pt() << std::endl
      << "    eta/phi: " << eta () << '/' << phi () << std::endl
       << "    # of constituents: " << nConstituents () << std::endl;
  out << "    Constituents:" << std::endl;
  for (unsigned index = 0; index < numberOfDaughters(); index++) {
    CandidateBaseRef constituent = daughterRef (index); // deref
    if (constituent.isNonnull()) {
      out << "      #" << index << " p/pt/eta/phi: " 
	  << constituent->p() << '/' << constituent->pt() << '/' << constituent->eta() << '/' << constituent->phi() 
	  << "    productId/index: " << constituent.id() << '/' << constituent.key() << std::endl; 
    }
    else {
      out << "      #" << index << " constituent is not available in the event"  << std::endl;
    }
  }
  return out.str ();
}

void Jet::scaleEnergy (double fScale) {
  setP4 (p4() * fScale);
}

bool Jet::isJet() const {
  return true;
}
