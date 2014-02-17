// Jet.cc
// Fedor Ratnikov, UMd
// $Id: Jet.cc,v 1.29 2012/02/01 14:51:11 pandolf Exp $

#include <sstream>
#include <cmath>

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

//Own header file
#include "DataFormats/JetReco/interface/Jet.h"



using namespace reco;

namespace {
  // approximate simple CALO geometry
  // abstract baseclass for geometry.

  class CaloPoint {
  public:
    static const double depth; // one for all relative depth of the reference point between ECAL begin and HCAL end
    static const double R_BARREL;
    static const double R_BARREL2;
    static const double Z_ENDCAP; // 1/2(EEz+HEz)
    static const double R_FORWARD; // eta=3
    static const double R_FORWARD2;
    static const double Z_FORWARD;
    static const double Z_BIG;
  };

  const double CaloPoint::depth = 0.1; // one for all relative depth of the reference point between ECAL begin and HCAL end
  const double CaloPoint::R_BARREL = (1.-depth)*143.+depth*407.;
  const double CaloPoint::R_BARREL2 = R_BARREL * R_BARREL;
  const double CaloPoint::Z_ENDCAP = (1.-depth)*320.+depth*568.; // 1/2(EEz+HEz)
  const double CaloPoint::R_FORWARD = Z_ENDCAP / std::sqrt (std::cosh(3.)*std::cosh(3.) -1.); // eta=3
  const double CaloPoint::R_FORWARD2 = R_FORWARD * R_FORWARD;
  const double CaloPoint::Z_FORWARD = 1100.+depth*165.;
  const double CaloPoint::Z_BIG = 1.e5;

  //old zvertex only implementation:
  class CaloPointZ: private CaloPoint{
  public:
    CaloPointZ (double fZ, double fEta){
      
      static const double ETA_MAX = 5.2;
      
      if (fZ > Z_ENDCAP) fZ = Z_ENDCAP-1.;
      if (fZ < -Z_ENDCAP) fZ = -Z_ENDCAP+1; // sanity check
      
      double tanThetaAbs = std::sqrt (std::cosh(fEta)*std::cosh(fEta) - 1.);
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
	else if (std::fabs (fEta) < ETA_MAX) zRef = Z_FORWARD; // forward
	
	mZ = fEta > 0 ? zRef : -zRef;
	mR = std::fabs ((mZ - fZ) / tanTheta);
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
    CaloPointZ(){};
    double mZ;
    double mR;
  };

  //new implementation to derive CaloPoint for free 3d vertex. 
  //code provided thanks to Christophe Saout
  template<typename Point>
  class CaloPoint3D : private CaloPoint {
  public:
    template<typename Vector, typename Point2>
    CaloPoint3D(const Point2 &vertex, const Vector &dir)
    {
      // note: no sanity checks here, make sure vertex is inside the detector!

      // check if positive or negative (or none) endcap should be tested
      int side = dir.z() < -1e-9 ? -1 : dir.z() > 1e-9 ? +1 : 0;

      double dirR = dir.Rho();

      // normalized direction in x-y plane
      double dirUnit[2] = { dir.x() / dirR, dir.y() / dirR };

      // rotate the vertex into a coordinate system where dir lies along x

      // vtxLong is the longitudinal coordinate of the vertex wrt/ dir
      double vtxLong = dirUnit[0] * vertex.x() + dirUnit[1] * vertex.y();

      // tIP is the (signed) transverse impact parameter
      double tIP = dirUnit[0] * vertex.y() - dirUnit[1] * vertex.x();

      // r and z coordinate
      double r, z;

      if (side) {
        double slope = dirR / dir.z();

        // check extrapolation to endcap
        r = vtxLong + slope * (side * Z_ENDCAP - vertex.z());
        double r2 = sqr(r) + sqr(tIP);

        if (r2 < R_FORWARD2) {
          // we are in the forward calorimeter, recompute
          r = vtxLong + slope * (side * Z_FORWARD - vertex.z());
          z = side * Z_FORWARD;
        } else if (r2 < R_BARREL2) {
          // we are in the endcap
          z = side * Z_ENDCAP;
        } else {
          // we are in the barrel, do the intersection below
          side = 0;
        }
      }

      if (!side) {
        // we are in the barrel
        double slope = dir.z() / dirR;
        r = std::sqrt(R_BARREL2 - sqr(tIP));
        z = vertex.z() + slope * (r - vtxLong);
      }

      // rotate (r, tIP, z) back into original x-y coordinate system
      point = Point(dirUnit[0] * r - dirUnit[1] * tIP,
                    dirUnit[1] * r + dirUnit[0] * tIP,
                    z);
    }

    const Point &caloPoint() const { return point; }

  private:
    template<typename T>
    static inline T sqr(const T &value) { return value * value; }

    Point point;
  };

}

Jet::Jet (const LorentzVector& fP4, 
	  const Point& fVertex, 
	  const Constituents& fConstituents)
  :  CompositePtrCandidate (0, fP4, fVertex),
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
  :  CompositePtrCandidate (0, fP4, fVertex),
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
// kept for backwards compatibility, use detector/physicsP4 instead!
float Jet::physicsEta (float fZVertex, float fDetectorEta) {
  CaloPointZ refPoint (0., fDetectorEta);
  return refPoint.etaReference (fZVertex);
}

/// static function to convert physics eta to detector eta
// kept for backwards compatibility, use detector/physicsP4 instead!
float Jet::detectorEta (float fZVertex, float fPhysicsEta) {
  CaloPointZ refPoint (fZVertex, fPhysicsEta);
  return refPoint.etaReference (0.);
}

Candidate::LorentzVector Jet::physicsP4 (const Candidate::Point &newVertex, const Candidate &inParticle,const Candidate::Point &oldVertex) {
  CaloPoint3D<Point> caloPoint(oldVertex,inParticle.momentum()); // Jet position in Calo.
  Vector physicsDir = caloPoint.caloPoint() - newVertex;
  double p = inParticle.momentum().r();
  Vector p3 = p * physicsDir.unit();
  LorentzVector returnVector(p3.x(), p3.y(), p3.z(), inParticle.energy());
  return returnVector;
}

Candidate::LorentzVector Jet::detectorP4 (const Candidate::Point &vertex, const Candidate &inParticle) {
  CaloPoint3D<Point> caloPoint(vertex,inParticle.momentum()); // Jet position in Calo.
  static const Point np(0,0,0);
  Vector detectorDir = caloPoint.caloPoint() - np;
  double p = inParticle.momentum().r();
  Vector p3 = p * detectorDir.unit();
  LorentzVector returnVector(p3.x(), p3.y(), p3.z(), inParticle.energy());
  return returnVector;
}


Jet::Constituents Jet::getJetConstituents () const {
  Jet::Constituents result;
  for (unsigned i = 0; i < numberOfDaughters(); i++) {
    result.push_back (daughterPtr (i));
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



float Jet::constituentPtDistribution() const {

  Jet::Constituents constituents = this->getJetConstituents();

  float sum_pt2 = 0.;
  float sum_pt  = 0.;

  for( unsigned iConst=0; iConst<constituents.size(); ++iConst ) {

    float pt = constituents[iConst]->p4().Pt();
    float pt2 = pt*pt;

    sum_pt += pt;
    sum_pt2 += pt2;

  } //for constituents

  float ptD_value = (sum_pt>0.) ? sqrt( sum_pt2 / (sum_pt*sum_pt) ) : 0.;

  return ptD_value;

} //constituentPtDistribution



float Jet::constituentEtaPhiSpread() const {

  Jet::Constituents constituents = this->getJetConstituents();


  float sum_pt2 = 0.;
  float sum_pt2deltaR2 = 0.;

  for( unsigned iConst=0; iConst<constituents.size(); ++iConst ) {

    LorentzVector thisConstituent = constituents[iConst]->p4();

    float pt = thisConstituent.Pt();
    float pt2 = pt*pt;
    double dR = deltaR (*this, *(constituents[iConst]));
    float pt2deltaR2 = pt*pt*dR*dR;

    sum_pt2 += pt2;
    sum_pt2deltaR2 += pt2deltaR2;

  } //for constituents

  float rmsCand_value = (sum_pt2>0.) ? sum_pt2deltaR2/sum_pt2 : 0.;

  return rmsCand_value;

} //constituentEtaPhiSpread



std::string Jet::print () const {
  std::ostringstream out;
  out << "Jet p/px/py/pz/pt: " << p() << '/' << px () << '/' << py() << '/' << pz() << '/' << pt() << std::endl
      << "    eta/phi: " << eta () << '/' << phi () << std::endl
       << "    # of constituents: " << nConstituents () << std::endl;
  out << "    Constituents:" << std::endl;
  for (unsigned index = 0; index < numberOfDaughters(); index++) {
    Constituent constituent = daughterPtr (index); // deref
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
