#include "TMath.h"
#include "../interface/MEzCalculator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void 
MEzCalculator::Print() 
{
  edm::LogInfo("Debug|MEzCalculator") << "[METzCalculator]: pxmu = " << lepton_.Px() << " pzmu= " << lepton_.Pz() << std::endl;
  edm::LogInfo("Debug|MEzCalculator") << "[METzCalculator]: pxnu = " << MET_.Px() << " pynu= " << MET_.Py() << std::endl;
}

MEzCalculator::MEzCalculator() 
{
  isComplex_ = false;
  isMuon_ = true;
}

MEzCalculator::~MEzCalculator() 
{
}

double
MEzCalculator::Calculate(int type) 
{  
  double M_W  = 80.4;
  double M_mu = 0.10566;
  double M_e  = 0.511e-3;
  double M_lepton = M_mu;
  if (!isMuon_) M_lepton = M_e;
  
  double emu = lepton_.Energy();
  double pxmu = lepton_.Px();
  double pymu = lepton_.Py();
  double pzmu = lepton_.Pz();
  double pxnu = MET_.Px();
  double pynu = MET_.Py();
  double pznu = 0.;
  
  double a = M_W*M_W - M_lepton*M_lepton + 2.0*pxmu*pxnu + 2.0*pymu*pynu;
  double A = 4.0*(emu*emu - pzmu*pzmu);
  double B = -4.0*a*pzmu;
  double C = 4.0*emu*emu*(pxnu*pxnu + pynu*pynu) - a*a;
  
  double tmproot = B*B - 4.0*A*C;
  
  if (tmproot<0) {
    isComplex_= true;
    pznu = - B/(2*A); // take real part of complex roots
  }
  else {
    isComplex_ = false;
    double tmpsol1 = (-B + TMath::Sqrt(tmproot))/(2.0*A);
    double tmpsol2 = (-B - TMath::Sqrt(tmproot))/(2.0*A);
    
    if (type == 0 ) {
      // two real roots, pick the one closest to pz of muon
      if (TMath::Abs(tmpsol2-pzmu) < TMath::Abs(tmpsol1-pzmu)) { pznu = tmpsol2;}
      else pznu = tmpsol1;
      // if pznu is > 300 pick the most central root
      if ( pznu > 300. ) {
	if (TMath::Abs(tmpsol1)<TMath::Abs(tmpsol2) ) pznu = tmpsol1;
	else pznu = tmpsol2;
      }
    }
    if (type == 1 ) {
      // two real roots, pick the one closest to pz of muon
      if (TMath::Abs(tmpsol2-pzmu) < TMath::Abs(tmpsol1-pzmu)) { pznu = tmpsol2;}
      else pznu = tmpsol1;
    }
    if (type == 2 ) {
      // pick the most central root.
      if (TMath::Abs(tmpsol1)<TMath::Abs(tmpsol2) ) pznu = tmpsol1;
      else pznu = tmpsol2;
    }
    if (type == 3 ) {
      // pick the largest value of the cosine
      TVector3 p3w, p3mu;
      p3w.SetXYZ(pxmu+pxnu, pymu+pynu, pzmu+ tmpsol1);
      p3mu.SetXYZ(pxmu, pymu, pzmu );
		
      double sinthcm1 = 2.*(p3mu.Perp(p3w))/M_W;
      p3w.SetXYZ(pxmu+pxnu, pymu+pynu, pzmu+ tmpsol2);
      double sinthcm2 = 2.*(p3mu.Perp(p3w))/M_W;
		
      double costhcm1 = TMath::Sqrt(1. - sinthcm1*sinthcm1);
      double costhcm2 = TMath::Sqrt(1. - sinthcm2*sinthcm2);
		
      if ( costhcm1 > costhcm2 ) pznu = tmpsol1;
      else pznu = tmpsol2;
    }
  }
  return pznu;
}
