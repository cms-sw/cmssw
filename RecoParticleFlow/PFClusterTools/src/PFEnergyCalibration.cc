#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"
#include <TMath.h>
#include <math.h>
#include <vector>
#include <TF1.h>
#include <map>
#include <algorithm>

using namespace std;

PFEnergyCalibration::PFEnergyCalibration() : pfCalibrations(0)
{
  initializeCalibrationFunctions();
}

PFEnergyCalibration::~PFEnergyCalibration() 
{

  delete faBarrel;
  delete fbBarrel;
  delete fcBarrel;
  delete faEtaBarrel;
  delete fbEtaBarrel;
  delete faEndcap;
  delete fbEndcap;
  delete fcEndcap;
  delete faEtaEndcap;
  delete fbEtaEndcap;

}

void
PFEnergyCalibration::initializeCalibrationFunctions() {

  // NEW NEW with HCAL pre-calibration

  threshE = 3.5;
  threshH = 2.5;

  // Barrel (fit made with |eta| < 1.2)
  faBarrel = new TF1("faBarrel","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",1.,1000.);
  fbBarrel = new TF1("fbBarrel","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",1.,1000.);
  fcBarrel = new TF1("fcBarrel","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",1.,1000.);
  faEtaBarrel = new TF1("faEtaBarrel","[0]+[1]*exp(-x/[2])",1.,1000.);
  fbEtaBarrel = new TF1("fbEtaBarrel","[0]+[1]*exp(-x/[2])+[3]*[3]*exp(-x*x/([4]*[4]))",1.,1000.);
  faBarrel->SetParameter(0,1.15665);
  fbBarrel->SetParameter(0,0.994603);
  fcBarrel->SetParameter(0,0.956544);
  faEtaBarrel->SetParameter(0,0.014664);
  fbEtaBarrel->SetParameter(0,0.00975451);
  faBarrel->SetParameter(1,0.165627);
  fbBarrel->SetParameter(1,0.13632);
  fcBarrel->SetParameter(1,0.0857207);
  faEtaBarrel->SetParameter(1,-0.0426776);
  fbEtaBarrel->SetParameter(1,0.102247);
  faBarrel->SetParameter(2,0.827718);
  fbBarrel->SetParameter(2,-0.758013);
  fcBarrel->SetParameter(2,-0.44347);
  faEtaBarrel->SetParameter(2,431.054);
  fbEtaBarrel->SetParameter(2,436.21);
  faBarrel->SetParameter(3,231.339);
  fbBarrel->SetParameter(3,183.627);
  fcBarrel->SetParameter(3,63.3479);
  faBarrel->SetParameter(4,2.45332);
  fbBarrel->SetParameter(4,1);
  fcBarrel->SetParameter(4,1.24174);
  faBarrel->SetParameter(5,29.6603);
  fbBarrel->SetParameter(5,39.6784);
  fcBarrel->SetParameter(5,12.322);

  // End-caps (fit made with eta 
  faEndcap = new TF1("faEndcap","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",1.,1000.);
  fbEndcap = new TF1("fbEndcap","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",1.,1000.);
  fcEndcap = new TF1("fcEndcap","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",1.,1000.);
  faEtaEndcap = new TF1("faEtaEndcap","[0]+[1]*exp(-x/[2])",1.,1000.);
  fbEtaEndcap = new TF1("fbEtaEndcap","[0]+[1]*exp(-x/[2])+[3]*[3]*exp(-x*x/([4]*[4]))",1.,1000.);
  faEndcap->SetParameter(0,1.1272);
  fbEndcap->SetParameter(0,0.982824);
  fcEndcap->SetParameter(0,0.950244);
  faEtaEndcap->SetParameter(0,-0.000582903);
  fbEtaEndcap->SetParameter(0,0.0267319);
  faEndcap->SetParameter(1,0.258536);
  fbEndcap->SetParameter(1,0.0977533);
  fcEndcap->SetParameter(1,0.00564779);
  faEtaEndcap->SetParameter(1,-0.000482148);
  fbEtaEndcap->SetParameter(1,-0.554552);
  faEndcap->SetParameter(2,0.808071);
  fbEndcap->SetParameter(2,0.155416);
  fcEndcap->SetParameter(2,0.227162);
  faEtaEndcap->SetParameter(2,209.466);
  fbEtaEndcap->SetParameter(2,1.71188);
  faEndcap->SetParameter(3,214.039);
  fbEndcap->SetParameter(3,240.379);
  fcEndcap->SetParameter(3,207.786);
  fbEtaEndcap->SetParameter(3,0.235834);
  faEndcap->SetParameter(4,2);
  fbEndcap->SetParameter(4,1.2);
  fcEndcap->SetParameter(4,1.32824);
  fbEtaEndcap->SetParameter(4,-135.431);
  faEndcap->SetParameter(5,47.2602);
  fbEndcap->SetParameter(5,78.3083);
  fcEndcap->SetParameter(5,22.1825);
  
}

void 
PFEnergyCalibration::energyEmHad(double t, double& e, double&h, double eta, double phi) const { 
 
  
  // Use calorimetric energy as true energy for neutral particles
  double tt = t;
  double ee = e;
  double hh = h;
  double a = 1.;
  double b = 1.;
  double etaCorrE = 1.;
  double etaCorrH = 1.;
  t = min(999.9,max(tt,e+h));
  if ( t < 1. ) return;

  // Barrel calibration
  if ( fabs(eta) < 1.48 ) { 

    // The energy correction
    a = e>0. ? aBarrel(t) : 1.;
    b = e>0. ? bBarrel(t) : cBarrel(t);
    double thresh = e > 0. ? threshE : threshH;

    // Protection against negative calibration - to be tuned
    if ( a < -0.25 || b < -0.25 ) { 
      a = 1.;
      b = 1.;
      thresh = 0.;
    }

    // The new estimate of the true energy
    t = min(999.9,max(tt, thresh+a*e+b*h));

    // The angular correction for ECAL hadronic deposits
    etaCorrE = 1. + aEtaBarrel(t) + bEtaBarrel(t)*fabs(eta)*fabs(eta);
    etaCorrH = 1.;
    // etaCorr = 1.;
    t = max(tt, thresh+etaCorrE*a*e+etaCorrH*b*h);

    if ( e > 0. && thresh > 0. ) 
      e = h > 0. ? threshE-threshH + etaCorrE * a * e : threshE + etaCorrE * a * e;
    if ( h > 0. && thresh > 0. ) 
      h = threshH + etaCorrH * b * h;

    /*
    if ( e < 0. || h < 0. ) { 
      std::cout << "Warning : Energy correction ! " << std::endl
		<< "eta,tt,e,h,a,b = " << eta << " " << tt << " " 
		<< ee << "/" << e << " " << hh << "/" << h << " " << a << " " << b << std::endl;
    }
      
    if ( etaCorrE > 2. || etaCorrE < 0.5 || 
	 etaCorrH > 2. || etaCorrH < 0.5 ) 
      std::cout << "Warning : Angular correction ! " << std::endl
		<< "etaCorrE,etaCorrH,eta,t = " 
		<< etaCorrE << " " << etaCorrH << " " << eta << " " << t << std::endl;
    */

  // Endcap calibration   
  } else {

    // The energy correction
    a = e>0. ? aEndcap(t) : 1.;
    b = e>0. ? bEndcap(t) : cEndcap(t);
    double thresh = e > 0. ? threshE : threshH;

    if ( a < -0.25 || b < -0.25 ) { 
      a = 1.;
      b = 1.;
      thresh = 0.;
    }

    // The new estimate of the true energy
    t = min(999.9,max(tt, thresh+a*e+b*h));
    
    // The angular correction
    double dEta = fabs ( fabs(eta) - 1.5 );
    double etaPow = dEta * dEta * dEta * dEta;
    //etaCorrE = 1. + aEtaEndcap(t) + 0.5*bEtaEndcap(t)*etaPow;
    etaCorrE = 1. + aEtaEndcap(t) + bEtaEndcap(t)*etaPow;
    etaCorrH = 1. + aEtaEndcap(t) + bEtaEndcap(t)*etaPow;
    /*
    if ( etaCorr > 2. || etaCorr < 0.5 ) 
      std::cout << "Warning : Angular correction ! " << std::endl
		<< "etaCorr,eta,t = " << etaCorr << " " << eta << " " << tt 
		<< " ee,hh,e,h = " << e << " " << h << " " << a*e << " " << b*h  
		<< std::endl;
    */

    t = min(999.9,max(tt, thresh + etaCorrE*a*e + etaCorrH*b*h));

    if ( e > 0. && thresh > 0. ) 
      e = h > 0. ? threshE-threshH + etaCorrE * a * e : threshE + etaCorrE * a * e;
    if ( h > 0. && thresh > 0. ) 
      h = threshH + b * etaCorrH * h;


  }

  // Protection
  if ( e < 0. || h < 0. ) {
    /*
    std::cout << "Warning : Energy correction ! " << std::endl
	      << "eta,tt,e,h,a,b = " << eta << " " << tt << " " 
	      << ee << "/" << e << " " << hh << "/" << h << " " << a << " " << b << std::endl;
    */
    // Some protection against crazy calibration
    if ( e < 0. ) e = ee;
    if ( h < 0. ) h = hh;
  }

  // And that's it !

  
}

// The calibration functions
double 
PFEnergyCalibration::aBarrel(double x) const { 

  if ( pfCalibrations ) { 

    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfa_BARREL,point); 

  } else { 

    return faBarrel->Eval(x); 

  }
}

double 
PFEnergyCalibration::bBarrel(double x) const { 

  if ( pfCalibrations ) { 

    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfb_BARREL,point); 

  } else { 

    return fbBarrel->Eval(x); 

  }
}

double 
PFEnergyCalibration::cBarrel(double x) const { 

  if ( pfCalibrations ) { 

    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfc_BARREL,point); 

  } else { 

    return fcBarrel->Eval(x); 

  }
}

double 
PFEnergyCalibration::aEtaBarrel(double x) const { 

  if ( pfCalibrations ) { 

    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfaEta_BARREL,point); 

  } else { 

    return faEtaBarrel->Eval(x); 

  }
}

double 
PFEnergyCalibration::bEtaBarrel(double x) const { 

  if ( pfCalibrations ) { 

    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfbEta_BARREL,point); 

  } else { 

    return fbEtaBarrel->Eval(x); 

  }
}

double 
PFEnergyCalibration::aEndcap(double x) const { 

  if ( pfCalibrations ) { 

    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfa_ENDCAP,point); 

  } else { 

    return faEndcap->Eval(x); 

  }
}

double 
PFEnergyCalibration::bEndcap(double x) const { 

  if ( pfCalibrations ) { 

    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfb_ENDCAP,point); 

  } else { 

    return fbEndcap->Eval(x); 

  }
}

double 
PFEnergyCalibration::cEndcap(double x) const { 

  if ( pfCalibrations ) { 

    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfc_ENDCAP,point); 

  } else { 

    return fcEndcap->Eval(x); 

  }
}

double 
PFEnergyCalibration::aEtaEndcap(double x) const { 

  if ( pfCalibrations ) { 

    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfaEta_ENDCAP,point); 

  } else { 

    return faEtaEndcap->Eval(x); 

  }
}

double 
PFEnergyCalibration::bEtaEndcap(double x) const { 

  if ( pfCalibrations ) { 

    BinningPointByMap point;
    point.insert(BinningVariables::JetEt, x);
    return pfCalibrations->getResult(PerformanceResult::PFfbEta_ENDCAP,point); 

  } else { 

    return fbEtaEndcap->Eval(x); 

  }
}


double
PFEnergyCalibration::energyEm(const reco::PFCluster& clusterEcal,
			      std::vector<double> &EclustersPS1,
			      std::vector<double> &EclustersPS2,
			      bool crackCorrection ){
  double ePS1(std::accumulate(EclustersPS1.begin(), EclustersPS1.end(), 0.d));
  double ePS2(std::accumulate(EclustersPS2.begin(), EclustersPS2.end(), 0.d));
  return energyEm(clusterEcal, ePS1, ePS2, crackCorrection);
}

double
PFEnergyCalibration::energyEm(const reco::PFCluster& clusterEcal,
			      double ePS1,
			      double ePS2,
			      bool crackCorrection ){
  double eEcal = clusterEcal.energy();
  //temporaty ugly fix
  reco::PFCluster myPFCluster=clusterEcal;
  myPFCluster.calculatePositionREP();
  double eta = myPFCluster.positionREP().eta();
  double phi = myPFCluster.positionREP().phi();

  double calibrated = Ecorr(eEcal,ePS1,ePS2,eta,phi, crackCorrection);
  if(eEcal!=0 && calibrated==0) std::cout<<"Eecal = "<<eEcal<<"  eta = "<<eta<<"  phi = "<<phi<<std::endl; 
  return calibrated; 
}

double PFEnergyCalibration::energyEm(const reco::PFCluster& clusterEcal,
				     std::vector<double> &EclustersPS1,
				     std::vector<double> &EclustersPS2,
				     double& ps1,double& ps2,
				     bool crackCorrection){
  double ePS1(std::accumulate(EclustersPS1.begin(), EclustersPS1.end(), 0.d));
  double ePS2(std::accumulate(EclustersPS2.begin(), EclustersPS2.end(), 0.d));
  return energyEm(clusterEcal, ePS1, ePS2, ps1, ps2, crackCorrection);
}
double PFEnergyCalibration::energyEm(const reco::PFCluster& clusterEcal,
				     double ePS1, double ePS2,
				     double& ps1,double& ps2,
				     bool crackCorrection){
  double eEcal = clusterEcal.energy();
  //temporaty ugly fix
  reco::PFCluster myPFCluster=clusterEcal;
  myPFCluster.calculatePositionREP();
  double eta = myPFCluster.positionREP().eta();
  double phi = myPFCluster.positionREP().phi();

  double calibrated = Ecorr(eEcal,ePS1,ePS2,eta,phi,ps1,ps2,crackCorrection);
  if(eEcal!=0 && calibrated==0) std::cout<<"Eecal = "<<eEcal<<"  eta = "<<eta<<"  phi = "<<phi<<std::endl; 
  return calibrated; 
}


std::ostream& operator<<(std::ostream& out,
			 const PFEnergyCalibration& calib) {

  if(!out ) return out;

  out<<"PFEnergyCalibration -- "<<endl;

  if ( calib.pfCalibrations ) { 

    std::cout << "Functions taken from the global tags : " << std::endl;

    static std::map<std::string, PerformanceResult::ResultType> functType;

    functType["PFfa_BARREL"] = PerformanceResult::PFfa_BARREL;
    functType["PFfa_ENDCAP"] = PerformanceResult::PFfa_ENDCAP;
    functType["PFfb_BARREL"] = PerformanceResult::PFfb_BARREL;
    functType["PFfb_ENDCAP"] = PerformanceResult::PFfb_ENDCAP;
    functType["PFfc_BARREL"] = PerformanceResult::PFfc_BARREL;
    functType["PFfc_ENDCAP"] = PerformanceResult::PFfc_ENDCAP;
    functType["PFfaEta_BARREL"] = PerformanceResult::PFfaEta_BARREL;
    functType["PFfaEta_ENDCAP"] = PerformanceResult::PFfaEta_ENDCAP;
    functType["PFfbEta_BARREL"] = PerformanceResult::PFfbEta_BARREL;
    functType["PFfbEta_ENDCAP"] = PerformanceResult::PFfbEta_ENDCAP;
    
    for(std::map<std::string,PerformanceResult::ResultType>::const_iterator 
	  func = functType.begin(); 
        func != functType.end(); 
        ++func) {    
      
      cout << "Function: " << func->first << endl;
      PerformanceResult::ResultType fType = func->second;
      calib.pfCalibrations->printFormula(fType);
    }

  } else { 
    
    std::cout << "Default calibration functions : " << std::endl;
    
    calib.faBarrel->Print();
    calib.fbBarrel->Print();
    calib.fcBarrel->Print();
    calib.faEtaBarrel->Print();
    calib.fbEtaBarrel->Print();
    calib.faEndcap->Print();
    calib.fbEndcap->Print();
    calib.fcEndcap->Print();
    calib.faEtaEndcap->Print();
    calib.fbEtaEndcap->Print();
  }
    
  return out;
}




///////////////////////////////////////////////////////////////
////                                                       ////  
////             CORRECTION OF PHOTONS' ENERGY             ////
////                                                       ////
////              Material effect: No tracker              ////
////       Tuned on CMSSW_2_1_0_pre4, Full Sim events      ////
////                                                       ////
///////////////////////////////////////////////////////////////
////                                                       ////
////            Jonathan Biteau - June 2008                ////
////                                                       ////
///////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////
////                                                       ////  
////  USEFUL FUNCTIONS FOR THE CORRECTION IN THE BARREL    ////
////                                                       ////
///////////////////////////////////////////////////////////////


//useful to compute the signed distance to the closest crack in the barrel
double
PFEnergyCalibration::minimum(double a,double b){
  if(TMath::Abs(b)<TMath::Abs(a)) a=b;
  return a;
}


//compute the unsigned distance to the closest phi-crack in the barrel
double
PFEnergyCalibration::dCrackPhi(double phi, double eta){

  constexpr double pi= M_PI;// 3.14159265358979323846;
  
  //Location of the 18 phi-cracks
  static std::vector<double> cPhi;
  if(cPhi.size()==0)
    {
      cPhi.resize(18,0);
      cPhi[0]=2.97025;
      for(unsigned i=1;i<=17;++i) cPhi[i]=cPhi[0]-2*i*pi/18;
    }

  //Shift of this location if eta<0
  constexpr double delta_cPhi=0.00638;

  double m; //the result

  //the location is shifted
  if(eta<0) phi +=delta_cPhi;

  if (phi>=-pi && phi<=pi){

    //the problem of the extrema
    if (phi<cPhi[17] || phi>=cPhi[0]){
      if (phi<0) phi+= 2*pi;
      m = minimum(phi -cPhi[0],phi-cPhi[17]-2*pi);        	
    }

    //between these extrema...
    else{
      bool OK = false;
      unsigned i=16;
      while(!OK){
	if (phi<cPhi[i]){
	  m=minimum(phi-cPhi[i+1],phi-cPhi[i]);
	  OK=true;
	}
	else i-=1;
      }
    }
  }
  else{
    m=0.;        //if there is a problem, we assum that we are in a crack
    std::cout<<"Problem in dminphi"<<std::endl;
  }
  if(eta<0) m=-m;   //because of the disymetry
  return m;
}

// corrects the effect of phi-cracks
double
PFEnergyCalibration::CorrPhi(double phi, double eta) {

  // we use 3 gaussians to correct the phi-cracks effect
  constexpr double p1=   5.59379e-01;
  constexpr double p2=   -1.26607e-03;
  constexpr double p3=  9.61133e-04;

  constexpr double p4=   1.81691e-01;
  constexpr double p5=   -4.97535e-03;
  constexpr double p6=   1.31006e-03;

  constexpr double p7=   1.38498e-01;
  constexpr double p8=   1.18599e-04;
  constexpr double p9= 2.01858e-03;
  

  double dminphi = dCrackPhi(phi,eta);
  
  double result = (1+p1*TMath::Gaus(dminphi,p2,p3)+p4*TMath::Gaus(dminphi,p5,p6)+p7*TMath::Gaus(dminphi,p8,p9));

  return result;
}   


// corrects the effect of  |eta|-cracks
double
PFEnergyCalibration::CorrEta(double eta){
  
  // we use a gaussian with a screwness for each of the 5 |eta|-cracks
  constexpr double a[] = {6.13349e-01, 5.08146e-01, 4.44480e-01, 3.3487e-01, 7.65627e-01}; // amplitude
  constexpr double m[] = {-1.79514e-02, 4.44747e-01, 7.92824e-01, 1.14090e+00, 1.47464e+00}; // mean
  constexpr double s[] = {7.92382e-03, 3.06028e-03, 3.36139e-03, 3.94521e-03, 8.63950e-04}; // sigma
  constexpr double sa[] = {1.27228e+01, 3.81517e-02, 1.63507e-01, -6.56480e-02, 1.87160e-01}; // screwness amplitude
  constexpr double ss[] = {5.48753e-02, -1.00223e-02, 2.22866e-03, 4.26288e-04, 2.67937e-03}; // screwness sigma
  double result = 1;

  for(unsigned i=0;i<=4;i++) result+=a[i]*TMath::Gaus(eta,m[i],s[i])*(1+sa[i]*TMath::Sign(1.,eta-m[i])*TMath::Exp(-TMath::Abs(eta-m[i])/ss[i]));

  return result;
}


//corrects the global behaviour in the barrel
double
PFEnergyCalibration::CorrBarrel(double E, double eta) {

  //Energy dependency
  /*
  //YM Parameters 52XX:
  constexpr double p0=1.00000e+00;
  constexpr double p1=3.27753e+01;
  constexpr double p2=2.28552e-02;
  constexpr double p3=3.06139e+00;
  constexpr double p4=2.25135e-01;
  constexpr double p5=1.47824e+00;
  constexpr double p6=1.09e-02;
  constexpr double p7=4.19343e+01;
  */
  constexpr double p0 = 0.9944;
  constexpr double p1 = 9.827;
  constexpr double p2 = 1.503;
  constexpr double p3 = 1.196;
  constexpr double p4 = 0.3349;
  constexpr double p5 = 0.89;
  constexpr double p6 = 0.004361;
  constexpr double p7 = 51.51;
  //Eta dependency
  constexpr double p8=2.705593e-03;
  
  double result = (p0+1/(p1+p2*TMath::Power(E,p3))+p4*TMath::Exp(-E/p5)+p6*TMath::Exp(-E*E/(p7*p7)))*(1+p8*eta*eta);

  return result;
}



///////////////////////////////////////////////////////////////
////                                                       ////  
////  USEFUL FUNCTIONS FOR THE CORRECTION IN THE ENDCAPS   ////
////  Parameters tuned for:                                ////
////          dR(ClustersPS1,ClusterEcal) < 0.08           ////
////          dR(ClustersPS2,ClusterEcal) < 0.13           ////
////                                                       ////
///////////////////////////////////////////////////////////////


//Alpha, Beta, Gamma give the weight of each sub-detector (PS layer1, PS layer2 and Ecal) in the areas of the endcaps where there is a PS
// Etot = Beta*eEcal + Gamma*(ePS1 + Alpha*ePS2) 

double
PFEnergyCalibration::Alpha(double eta) {

  //Energy dependency
  constexpr double p0 = 5.97621e-01;

  //Eta dependency
  constexpr double p1 =-1.86407e-01;
  constexpr double p2 = 3.85197e-01; 

  //so that <feta()> = 1
  constexpr double norm = (p1+p2*(2.6+1.656)/2);

  double result = p0*(p1+p2*eta)/norm;

  return result;
}

double
PFEnergyCalibration::Beta(double E, double eta) {

 //Energy dependency
  constexpr double p0 = 0.032;
  constexpr double p1 = 9.70394e-02;
  constexpr double p2 = 2.23072e+01;
  constexpr double p3 = 100;

  //Eta dependency
  constexpr double p4 = 1.02496e+00 ;
  constexpr double p5 = -4.40176e-03 ;

  //so that <feta()> = 1
  constexpr double norm = (p4+p5*(2.6+1.656)/2);

  double result = (1.0012+p0*TMath::Exp(-E/p3)+p1*TMath::Exp(-E/p2))*(p4+p5*eta)/norm;			  
  return result;
}


double
PFEnergyCalibration::Gamma(double etaEcal) {

 //Energy dependency
  constexpr double p0 = 2.49752e-02;

  //Eta dependency
  constexpr double p1 = 6.48816e-02;
  constexpr double p2 = -1.59517e-02; 
 
  //so that <feta()> = 1
  constexpr double norm = (p1+p2*(2.6+1.656)/2);

  double result = p0*(p1+p2*etaEcal)/norm;					  

  return result;
}



///////////////////////////////////////////////////////////////
////                                                       ////  
////   THE CORRECTIONS IN THE BARREL AND IN THE ENDCAPS    ////
////                                                       ////
///////////////////////////////////////////////////////////////


// returns the corrected energy in the barrel (0,1.48)
// Global Behaviour, phi and eta cracks are taken into account
double
PFEnergyCalibration::EcorrBarrel(double E, double eta, double phi,
				 bool crackCorrection ){

  // double result = E*CorrBarrel(E,eta)*CorrEta(eta)*CorrPhi(phi,eta);
  double correction = crackCorrection ? std::max(CorrEta(eta),CorrPhi(phi,eta)) : 1.;
  double result = E * CorrBarrel(E,eta) * correction;

  return result;
}


// returns the corrected energy in the area between the barrel and the PS (1.48,1.65)
double
PFEnergyCalibration::EcorrZoneBeforePS(double E, double eta){

 //Energy dependency
  constexpr double p0 =1; 
  constexpr double p1 =0.18;
  constexpr double p2 =8.;

  //Eta dependency
  constexpr double p3 =0.3;
  constexpr double p4 =1.11;
  constexpr double p5 =0.025;
  constexpr double p6 =1.49;
  constexpr double p7 =0.6;

  //so that <feta()> = 1
  constexpr double norm = 1.21;

  double result = E*(p0+p1*TMath::Exp(-E/p2))*(p3+p4*TMath::Gaus(eta,p6,p5)+p7*eta)/norm;

  return result;
}


// returns the corrected energy in the PS (1.65,2.6)
// only when (ePS1>0)||(ePS2>0)
double
PFEnergyCalibration::EcorrPS(double eEcal,double ePS1,double ePS2,double etaEcal) {

  // gives the good weights to each subdetector
  double E = Beta(1.0155*eEcal+0.025*(ePS1+0.5976*ePS2)/9e-5,etaEcal)*eEcal+Gamma(etaEcal)*(ePS1+Alpha(etaEcal)*ePS2)/9e-5 ;

  //Correction of the residual energy dependency
  constexpr double p0 = 1.00;
  constexpr double p1 = 2.18;
  constexpr double p2 =1.94;
  constexpr double p3 =4.13;
  constexpr double p4 =1.127;

  double result = E*(p0+p1*TMath::Exp(-E/p2)-p3*TMath::Exp(-E/p4));

  return result;
} 

// returns the corrected energy in the PS (1.65,2.6)
// only when (ePS1>0)||(ePS2>0)
double
PFEnergyCalibration::EcorrPS(double eEcal,double ePS1,double ePS2,double etaEcal,double & outputPS1, double & outputPS2) {

  // gives the good weights to each subdetector
  double gammaprime=Gamma(etaEcal)/9e-5;
  outputPS1=gammaprime*ePS1;
  outputPS2=gammaprime*Alpha(etaEcal)*ePS2;
  double E = Beta(1.0155*eEcal+0.025*(ePS1+0.5976*ePS2)/9e-5,etaEcal)*eEcal+outputPS1+outputPS2;

  //Correction of the residual energy dependency
  constexpr double p0 = 1.00;
  constexpr double p1 = 2.18;
  constexpr double p2 =1.94;
  constexpr double p3 =4.13;
  constexpr double p4 =1.127;
  
  double corrfac=(p0+p1*TMath::Exp(-E/p2)-p3*TMath::Exp(-E/p4));
  outputPS1*=corrfac;
  outputPS2*=corrfac;
  double result = E*corrfac;

  return result;
} 


// returns the corrected energy in the PS (1.65,2.6)
// only when (ePS1=0)&&(ePS2=0)
double 
PFEnergyCalibration::EcorrPS_ePSNil(double eEcal,double eta){

  //Energy dependency
  constexpr double p0= 1.02;
  constexpr double p1= 0.165;
  constexpr double p2= 6.5 ;
  constexpr double p3=  2.1 ;

  //Eta dependency
  constexpr double p4 = 1.02496e+00 ;
  constexpr double p5 = -4.40176e-03 ;

  //so that <feta()> = 1
  constexpr double norm = (p4+p5*(2.6+1.656)/2);

  double result = eEcal*(p0+p1*TMath::Exp(-TMath::Abs(eEcal-p3)/p2))*(p4+p5*eta)/norm;
		  
  return result;
}


// returns the corrected energy in the area between the end of the PS and the end of the endcap (2.6,2.98)
double
PFEnergyCalibration::EcorrZoneAfterPS(double E, double eta){

  //Energy dependency
  constexpr double p0 =1; 
  constexpr double p1 = 0.058;
  constexpr double p2 =12.5;
  constexpr double p3 =-1.05444e+00;
  constexpr double p4 =-5.39557e+00;
  constexpr double p5 =8.38444e+00;
  constexpr double p6 = 6.10998e-01  ;

  //Eta dependency
  constexpr double p7 =1.06161e+00;
  constexpr double p8 = 0.41;
  constexpr double p9 =2.918;
  constexpr double p10 =0.0181;
  constexpr double p11= 2.05;
  constexpr double p12 =2.99;
  constexpr double p13=0.0287;

  //so that <feta()> = 1
  constexpr double norm=1.045;

  double result = E*(p0+p1*TMath::Exp(-(E-p3)/p2)+1/(p4+p5*TMath::Power(E,p6)))*(p7+p8*TMath::Gaus(eta,p9,p10)+p11*TMath::Gaus(eta,p12,p13))/norm;
  return result;
}




// returns the corrected energy everywhere
// this work should be improved between 1.479 and 1.52 (junction barrel-endcap)
double
PFEnergyCalibration::Ecorr(double eEcal,double ePS1,double ePS2,
			   double eta,double phi,
			   bool crackCorrection ) {

  constexpr double endBarrel=1.48;
  constexpr double beginingPS=1.65;
  constexpr double endPS=2.6;
  constexpr double endEndCap=2.98;
 
  double result=0;

  eta=TMath::Abs(eta);

  if(eEcal>0){
    if(eta <= endBarrel)                         result = EcorrBarrel(eEcal,eta,phi,crackCorrection);
    else if(eta <= beginingPS)                   result = EcorrZoneBeforePS(eEcal,eta);
    else if((eta < endPS) && ePS1==0 && ePS2==0) result = EcorrPS_ePSNil(eEcal,eta);
    else if(eta < endPS)                         result = EcorrPS(eEcal,ePS1,ePS2,eta);
    else if(eta < endEndCap)                     result = EcorrZoneAfterPS(eEcal,eta); 
    else result =eEcal;
  }
  else result = eEcal;// useful if eEcal=0 or eta>2.98
  //protection
  if(result<eEcal) result=eEcal;
  return result;
}

// returns the corrected energy everywhere
// this work should be improved between 1.479 and 1.52 (junction barrel-endcap)
double
PFEnergyCalibration::Ecorr(double eEcal,double ePS1,double ePS2,double eta,double phi,double& ps1,double&ps2,bool crackCorrection)  {

  constexpr double endBarrel=1.48;
  constexpr double beginingPS=1.65;
  constexpr double endPS=2.6;
  constexpr double endEndCap=2.98;
 
  double result=0;

  eta=TMath::Abs(eta);

  if(eEcal>0){
    if(eta <= endBarrel)                         result = EcorrBarrel(eEcal,eta,phi,crackCorrection);
    else if(eta <= beginingPS)                   result = EcorrZoneBeforePS(eEcal,eta);
    else if((eta < endPS) && ePS1==0 && ePS2==0) result = EcorrPS_ePSNil(eEcal,eta);
    else if(eta < endPS)                         result = EcorrPS(eEcal,ePS1,ePS2,eta,ps1,ps2);
    else if(eta < endEndCap)                     result = EcorrZoneAfterPS(eEcal,eta); 
    else result =eEcal;
  }
  else result = eEcal;// useful if eEcal=0 or eta>2.98
  // protection
  if(result<eEcal) result=eEcal;
  return result;
}
