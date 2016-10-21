#include "Geometry/HGCalCommonData/interface/FastTimeDDDConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define EDM_ML_DEBUG

FastTimeDDDConstants::FastTimeDDDConstants(const FastTimeParameters* ft) : ftpar_(ft) {

#ifdef EDM_ML_DEBUG
  std::cout << "FastTimeDDDConstants::FastTimeDDDConstants ( const FastTimeParameters* ft ) constructor\n";
#endif
  initialize();

}

FastTimeDDDConstants::~FastTimeDDDConstants() { 
#ifdef EDM_ML_DEBUG
  std::cout << "FastTimeDDDConstants:destructed!!!" << std::endl;
#endif
}

std::pair<int,int> FastTimeDDDConstants::getZPhi(G4ThreeVector local) const {

  double z   = std::abs(local.z());
  double phi = local.phi();
  if (phi < 0) phi += CLHEP::twopi;
  int    iz   = (int)(z/dZBarrel_) + 1;
  if (iz   > ftpar_->nZBarrel_) iz    = ftpar_->nZBarrel_;
  int    iphi = (int)(phi/dPhiBarrel_) + 1;
  if (iphi > ftpar_->nPhiBarrel_) iphi = 1;
#ifdef EDM_ML_DEBUG
  std::cout << "FastTimeDDDConstants:Barrel z|phi " << z << " " 
	    << phi/CLHEP::deg << " iz|iphi " << iz << " " << iphi << std::endl;
#endif
  return std::pair<int,int>(iz,iphi);
}

std::pair<int,int> FastTimeDDDConstants::getEtaPhi(G4ThreeVector local) const {

  double r   = local.perp();
  double phi = local.phi();
  if (phi < 0) phi += CLHEP::twopi;
  int    ir(ftpar_->nEtaEndcap_);
  for (unsigned int k=1; k<rLimits_.size(); ++k) {
    if (r > rLimits_[k]) {
      ir    = k;  break;
    }
  }
  int    iphi = (int)(phi/dPhiEndcap_) + 1;
  if (iphi > ftpar_->nPhiEndcap_) iphi = 1;
#ifdef EDM_ML_DEBUG
  std::cout << "FastTimeDDDConstants:Endcap r|phi " << r << " " 
	    << phi/CLHEP::deg << " ir|iphi " << ir << " " << iphi << std::endl;
#endif
  return std::pair<int,int>(ir,iphi);
}

int FastTimeDDDConstants::getCells(int type) const {
  int numb(0);
  if (type == 1) {
    numb = (ftpar_->nZBarrel_)*(ftpar_->nPhiBarrel_);
  } else if (type == 2) {
    numb = (ftpar_->nEtaEndcap_)*(ftpar_->nPhiEndcap_);
  }
  return numb;
}

bool FastTimeDDDConstants::isValidXY(int type, int izeta, int iphi) const {
  bool ok(false);
  if (type == 1) {
    ok = ((izeta > 0) && (izeta <= ftpar_->nZBarrel_) && 
	  (iphi > 0) && (iphi <= ftpar_->nPhiBarrel_));
  } else if (type == 2) {
    ok = ((izeta > 0) && (izeta <= ftpar_->nEtaEndcap_) && 
	  (iphi > 0) && (iphi <= ftpar_->nPhiEndcap_));
  }
  return ok;
}

void FastTimeDDDConstants::initialize() {

  double thmin = atan(ftpar_->geomParEndcap_[0]/ftpar_->geomParEndcap_[2]);
  etaMax_      = -log(0.5*thmin);
  double thmax = atan(ftpar_->geomParEndcap_[1]/ftpar_->geomParEndcap_[2]);
  etaMin_      = -log(0.5*thmax);
  dEta_        = (etaMax_-etaMin_)/ftpar_->nEtaEndcap_;
#ifdef EDM_ML_DEBUG
  std::cout << "Theta range " << thmin/CLHEP::deg << ":" << thmax/CLHEP::deg
	    << " Eta range " << etaMin_ << ":" << etaMax_ << ":" << dEta_ 
	    << std::endl;
#endif
  for (int k=0; k<=ftpar_->nEtaEndcap_; ++k) {
    double eta   = etaMin_ + k*dEta_;
    double theta = 2.0*atan(exp(-eta));
    double rval  = (ftpar_->geomParEndcap_[2])*tan(theta);
    rLimits_.push_back(rval);
  }
  dZBarrel_   = ftpar_->geomParBarrel_[1]/ftpar_->nZBarrel_;
  dPhiBarrel_ = CLHEP::twopi/ftpar_->nPhiBarrel_;
  dPhiEndcap_ = CLHEP::twopi/ftpar_->nPhiEndcap_;
#ifdef EDM_ML_DEBUG
  std::cout << "FastTimeDDDConstants initialized with " << ftpar_->nZBarrel_ 
	    << ":" << ftpar_->nPhiBarrel_ << ":" << getCells(1) 
	    << " cells for barrel; dz|dphi " << dZBarrel_ << "|" << dPhiBarrel_
	    << " and " << ftpar_->nEtaEndcap_ << ":" << ftpar_->nPhiEndcap_
	    << ":" << getCells(2) << " cells for endcap; dphi " << dPhiEndcap_
	    << " The Limits in R are" << std::endl;
  for (unsigned int k=0; k<rLimits_.size(); ++k) {
    std::cout << "[" << k << "] " << rLimits_[k] << " ";
    if (k%8 == 7) std::cout << std::endl;
  }
  if ((rLimits_.size()-1)%8 != 7) std::cout << std::endl;
#endif
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(FastTimeDDDConstants);
