#include "Geometry/HcalCommonData/interface/HcalGeomParameters.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"
#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define EDM_ML_DEBUG

HcalGeomParameters::HcalGeomParameters() {
#ifdef EDM_ML_DEBUG
  std::cout << "HcalGeomParameters::HcalGeomParameters ( const DDCompactView& cpv ) constructor" << std::endl;
#endif
}

HcalGeomParameters::~HcalGeomParameters() { 
#ifdef EDM_ML_DEBUG
  std::cout << "HcalGeomParameters::destructed!!!" << std::endl;
#endif
}

void HcalGeomParameters::getConstRHO( std::vector<double>& rHO ) const {

  rHO.emplace_back(rminHO);
  for (double i : etaHO) rHO.emplace_back(i);
}

std::vector<int> HcalGeomParameters::getModHalfHBHE(const int type) const {

  std::vector<int> modHalf;
  if (type == 0) {
    modHalf.emplace_back(nmodHB); modHalf.emplace_back(nzHB);
  } else {
    modHalf.emplace_back(nmodHE); modHalf.emplace_back(nzHE);
  }
  return modHalf;
}

unsigned int HcalGeomParameters::find(int element, 
				      std::vector<int>& array) const {
  unsigned int id = array.size();
  for (unsigned int i = 0; i < array.size(); i++) {
    if (element == array[i]) {
      id = i;
      break;
    }
  }
  return id;
}
 
double HcalGeomParameters::getEta(double r, double z) const {

  double tmp = 0;
  if (z != 0) tmp = -log(tan(0.5*atan(r/z)));
#ifdef EDM_ML_DEBUG
  std::cout << "HcalGeomParameters::getEta " << r << " " << z  << " ==> " 
	    << tmp << std::endl;
#endif
  return tmp;
}

void HcalGeomParameters::loadGeometry(const DDFilteredView& _fv,  
				      HcalParameters& php) {

  DDFilteredView fv = _fv;
  bool dodet=true, hf=false;
  std::vector<double> rb(20,0.0), ze(20,0.0), thkb(20,-1.0), thke(20,-1.0);
  std::vector<int>    ib(20,0),   ie(20,0);
  std::vector<int>    izb, phib, ize, phie;
  std::vector<double> rxb;
#ifdef EDM_ML_DEBUG
  std::vector<double> rminHE(20,0.0), rmaxHE(20,0.0);
#endif
  php.rhoxHB.clear(); php.zxHB.clear(); php.dyHB.clear(); php.dxHB.clear();
  php.layHB.clear(); php.layHE.clear();
  php.zxHE.clear(); php.rhoxHE.clear(); php.dyHE.clear(); php.dx1HE.clear(); php.dx2HE.clear();
  dzVcal    = -1.;

  while (dodet) {
    DDTranslation    t    = fv.translation();
    std::vector<int> copy = fv.copyNumbers();
    const DDSolid & sol  = fv.logicalPart().solid();
    int idet = 0, lay = -1;
    int nsiz = (int)(copy.size());
    if (nsiz>0) lay  = copy[nsiz-1]/10;
    if (nsiz>1) idet = copy[nsiz-2]/1000;
    double dx=0, dy=0, dz=0, dx1=0, dx2=0;
#ifdef EDM_ML_DEBUG
    double alp(0);
#endif
    if (sol.shape() == DDSolidShape::ddbox) {
      const DDBox & box = static_cast<DDBox>(fv.logicalPart().solid());
      dx = box.halfX();
      dy = box.halfY();
      dz = box.halfZ();
    } else if (sol.shape() == DDSolidShape::ddtrap) {
      const DDTrap & trp = static_cast<DDTrap>(fv.logicalPart().solid());
      dx1= trp.x1();
      dx2= trp.x2();
      dx = 0.25*(trp.x1()+trp.x2()+trp.x3()+trp.x4());
      dy = 0.5*(trp.y1()+trp.y2());
      dz = trp.halfZ();
#ifdef EDM_ML_DEBUG
      alp= 0.5*(trp.alpha1()+trp.alpha2());
#endif
    } else if (sol.shape() == DDSolidShape::ddtubs) {
      const DDTubs & tub = static_cast<DDTubs>(fv.logicalPart().solid());
      dx = tub.rIn();
      dy = tub.rOut();
      dz = tub.zhalf();
    }
    if (idet == 3) {
      // HB
#ifdef EDM_ML_DEBUG
      std::cout << "HB " << sol.name() << " Shape " << sol.shape()
		<< " Layer " << lay << " R " << t.Rho() << std::endl;
#endif
      if (lay >=0 && lay < 20) {
	ib[lay]++;
	rb[lay] += t.Rho();
	if (thkb[lay] <= 0) {
	  if (lay < 17) thkb[lay] = dx;
	  else          thkb[lay] = std::min(dx,dy);
	}
	if (lay < 17) {
	  bool found = false;
	  for (double k : rxb) {
	    if (std::abs(k-t.Rho()) < 0.01) {
	      found = true;
	      break;
	    }
	  }
	  if (!found) {
	    rxb.emplace_back(t.Rho());
	    php.rhoxHB.emplace_back(t.Rho()*std::cos(t.phi()));
	    php.zxHB.emplace_back(std::abs(t.z()));
	    php.dyHB.emplace_back(2.*dy);
	    php.dxHB.emplace_back(2.*dz);
	    php.layHB.emplace_back(lay);
	  }
	}
      }
      if (lay == 2) {
	int iz = copy[nsiz-5];
	int fi = copy[nsiz-4];
	unsigned int it1 = find(iz, izb);
	if (it1 == izb.size())  izb.emplace_back(iz);
	unsigned int it2 = find(fi, phib);
	if (it2 == phib.size()) phib.emplace_back(fi);
      }
      if (lay == 18) {
	int ifi=-1, ich=-1;
	if (nsiz>2) ifi = copy[nsiz-3];
	if (nsiz>3) ich = copy[nsiz-4];
	double z1 = std::abs((t.z()) + dz);
	double z2 = std::abs((t.z()) - dz);
	if (std::abs(z1-z2) < 0.01) z1 = 0;
        if (ifi == 1 && ich == 4) {
	  if (z1 > z2) {
	    double tmp = z1;
	    z1 = z2;
	    z2 = tmp;
	  }
	  bool sok = true;
	  for (unsigned int kk=0; kk<php.zHO.size(); kk++) {
	    if (std::abs(z2-php.zHO[kk]) < 0.01) {
	      sok = false;
	      break;
	    }	else if (z2 < php.zHO[kk]) {
	      php.zHO.resize(php.zHO.size()+2);
	      for (unsigned int kz=php.zHO.size()-1; kz>kk+1; kz=kz-2) {
		php.zHO[kz]   = php.zHO[kz-2];
		php.zHO[kz-1] = php.zHO[kz-3];
	      }
	      php.zHO[kk+1] = z2;
	      php.zHO[kk]   = z1;
	      sok = false;
	      break;
	    }
	  }
	  if (sok) {
	    php.zHO.emplace_back(z1);
	    php.zHO.emplace_back(z2);
	  }
#ifdef EDM_ML_DEBUG
	  std::cout << "Detector " << idet << " Lay " << lay << " fi " << ifi 
		    << " " << ich << " z " << z1 << " " << z2 << std::endl;
#endif
	}
      }
    } else if (idet == 4) {
      // HE
#ifdef EDM_ML_DEBUG
      std::cout << "HE " << sol.name() << " Shape " << sol.shape()
		<< " Layer " << lay << " Z " << t.z() << std::endl;
#endif
      if (lay >=0 && lay < 20) {
	ie[lay]++;
	ze[lay] += std::abs(t.z());
	if (thke[lay] <= 0) thke[lay] = dz;
#ifdef EDM_ML_DEBUG
	double rinHE  = t.Rho()*cos(alp) - dy;
	double routHE = t.Rho()*cos(alp) + dy;
	rminHE[lay] += rinHE;
	rmaxHE[lay] += routHE;
#endif
	bool found = false;
	for (double k : php.zxHE) {
	  if (std::abs(k-std::abs(t.z())) < 0.01) {
	    found = true;
	    break;
	  }
	}
	if (!found) {
	  php.zxHE.emplace_back(std::abs(t.z()));
	  php.rhoxHE.emplace_back(t.Rho()*std::cos(t.phi()));
	  php.dyHE.emplace_back(dy*std::cos(t.phi()));
	  dx1 -= 0.5*(t.rho()-dy)*std::cos(t.phi())*std::tan(10*CLHEP::deg);
	  dx2 -= 0.5*(t.rho()+dy)*std::cos(t.phi())*std::tan(10*CLHEP::deg);
	  php.dx1HE.emplace_back(-dx1);
	  php.dx2HE.emplace_back(-dx2);
	  php.layHE.emplace_back(lay);
	}
      }
      if (copy[nsiz-1] == 21 || copy[nsiz-1] == 71) {
	int iz = copy[nsiz-7];
	int fi = copy[nsiz-5];
	unsigned int it1 = find(iz, ize);
	if (it1 == ize.size())  ize.emplace_back(iz);
	unsigned int it2 = find(fi, phie);
	if (it2 == phie.size()) phie.emplace_back(fi);
      }
    } else if (idet == 5) {
      // HF
      if (!hf) {
	const std::vector<double> & paras = sol.parameters();
#ifdef EDM_ML_DEBUG
	std::cout << "HF " << sol.name() << " Shape " << sol.shape()
		  << " Z " << t.z() << " with " << paras.size()
		  << " Parameters" << std::endl;
	for (unsigned j=0; j<paras.size(); j++)
	  std::cout << "HF Parameter[" << j << "] = " << paras[j] << std::endl;
#endif
	if (sol.shape() == DDSolidShape::ddpolycone_rrz) {
	  int nz  = (int)(paras.size())-3;
	  dzVcal  = 0.5*(paras[nz]-paras[3]);
	  hf      = true;
	} else if (sol.shape() == DDSolidShape::ddtubs || sol.shape() == DDSolidShape::ddcons) {
	  dzVcal  = paras[0];
	  hf      = true;
	}
      }
#ifdef EDM_ML_DEBUG
    } else {
      std::cout << "Unknown Detector " << idet << " for " << sol.name() 
		<< " Shape " << sol.shape() << " R " << t.Rho() << " Z " 
		<< t.z() << std::endl;
#endif
    }
    dodet = fv.next();
  }

  int ibmx = 0, iemx = 0;
  for (int i = 0; i < 20; i++) {
    if (ib[i]>0) {
      rb[i] /= (double)(ib[i]);
      ibmx   = i+1;
    }
    if (ie[i]>0) {
      ze[i] /= (double)(ie[i]);
      iemx   = i+1;
    }
#ifdef EDM_ML_DEBUG
    if (ie[i]> 0) {
      rminHE[i] /= (double)(ie[i]);
      rmaxHE[i] /= (double)(ie[i]);
    }
    std::cout << "Index " << i << " Barrel " << ib[i] << " "
	      << rb[i] << " Endcap " << ie[i] << " " << ze[i] << ":"
	      << rminHE[i] << ":" << rmaxHE[i] << std::endl;
#endif
  }
  for (int i = 4; i >= 0; i--) {
    if (ib[i] == 0) {rb[i] = rb[i+1]; thkb[i] = thkb[i+1];}
    if (ie[i] == 0) {ze[i] = ze[i+1]; thke[i] = thke[i+1];}
#ifdef EDM_ML_DEBUG
    if (ib[i] == 0 || ie[i] == 0)
      std::cout << "Index " << i << " Barrel " << ib[i] << " "
		<< rb[i] << " Endcap " << ie[i] << " " << ze[i] << std::endl;
#endif
  }

#ifdef EDM_ML_DEBUG
  for (unsigned int k=0; k<php.layHB.size(); ++k)
    std::cout << "HB: " << php.layHB[k] << " R " << rxb[k] << " " << php.rhoxHB[k] << " Z " << php.zxHB[k] << " DY " << php.dyHB[k] << " DZ " << php.dxHB[k] << "\n";
  for (unsigned int k=0; k<php.layHE.size(); ++k) 
    std::cout << "HE: " << php.layHE[k] << " R " << php.rhoxHE[k] << " Z " << php.zxHE[k] << " X1|X2 " << php.dx1HE[k] << "|" << php.dx2HE[k] << " DY " << php.dyHE[k] << "\n";
  std::cout << "HcalGeomParameters: Maximum Layer for HB " << ibmx << " for HE "
	    << iemx << " extent " << dzVcal << std::endl;
#endif

  if (ibmx > 0) {
    php.rHB.resize(ibmx);
    php.drHB.resize(ibmx);
    for (int i=0; i<ibmx; i++) {
      php.rHB[i]  = rb[i];
      php.drHB[i] = thkb[i];
#ifdef EDM_ML_DEBUG
      std::cout << "HcalGeomParameters: php.rHB[" << i << "] = " << php.rHB[i] 
		<< " php.drHB[" << i << "] = " << php.drHB[i] << std::endl;
#endif
    }
  }
  if (iemx > 0) {
    php.zHE.resize(iemx);
    php.dzHE.resize(iemx);
    for (int i=0; i<iemx; i++) {
      php.zHE[i]  = ze[i];
      php.dzHE[i] = thke[i];
#ifdef EDM_ML_DEBUG
      std::cout << "HcalGeomParameters: php.zHE[" << i << "] = " << php.zHE[i] 
		<< " php.dzHE[" << i << "] = " << php.dzHE[i] << std::endl;
#endif
    }
  }

  nzHB   = (int)(izb.size());
  nmodHB = (int)(phib.size());
#ifdef EDM_ML_DEBUG
  std::cout << "HcalGeomParameters::loadGeometry: " << nzHB
	    << " barrel half-sectors" << std::endl;
  for (int i=0; i<nzHB; i++)
    std::cout << "Section " << i << " Copy number " << izb[i] << std::endl;
  std::cout << "HcalGeomParameters::loadGeometry: " << nmodHB
	    << " barrel modules" << std::endl;
  for (int i=0; i<nmodHB; i++)
    std::cout << "Module " << i << " Copy number " << phib[i] << std::endl;
#endif

  nzHE   = (int)(ize.size());
  nmodHE = (int)(phie.size());
#ifdef EDM_ML_DEBUG
  std::cout << "HcalGeomParameters::loadGeometry: " << nzHE
	    << " endcap half-sectors" << std::endl;
  for (int i=0; i<nzHE; i++)
    std::cout << "Section " << i << " Copy number " << ize[i] << std::endl;
  std::cout << "HcalGeomParameters::loadGeometry: " << nmodHE
	    << " endcap modules" << std::endl;
  for (int i=0; i<nmodHE; i++)
    std::cout << "Module " << i << " Copy number " << phie[i] << std::endl;
#endif

#ifdef EDM_ML_DEBUG
  std::cout << "HO has Z of size " << php.zHO.size() << std::endl;
  for (unsigned int kk=0; kk<php.zHO.size(); kk++)
    std::cout << "ZHO[" << kk << "] = " << php.zHO[kk] << std::endl;
#endif
  if (ibmx > 17 && php.zHO.size() > 4) {
    rminHO   = php.rHB[17]-100.0;
    etaHO[0] = getEta(0.5*(php.rHB[17]+php.rHB[18]), php.zHO[1]);
    etaHO[1] = getEta(php.rHB[18]+php.drHB[18], php.zHO[2]);
    etaHO[2] = getEta(php.rHB[18]-php.drHB[18], php.zHO[3]);
    etaHO[3] = getEta(php.rHB[18]+php.drHB[18], php.zHO[4]);
  } else {
    rminHO   =-1.0;
    etaHO[0] = etaHO[1] = etaHO[2] = etaHO[3] = 0;
  }
#ifdef EDM_ML_DEBUG
  std::cout << "HO Eta boundaries " << etaHO[0] << " " << etaHO[1]
	    << " " << etaHO[2] << " " << etaHO[3] << std::endl;
  std::cout << "HO Parameters " << rminHO << " " << php.zHO.size();
  for (int i=0; i<4; ++i) std::cout << " eta[" << i << "] = " << etaHO[i];
  for (unsigned int i=0; i<php.zHO.size(); ++i) std::cout << " zho[" << i << "] = " << php.zHO[i];
  std::cout << std::endl;
#endif
}
