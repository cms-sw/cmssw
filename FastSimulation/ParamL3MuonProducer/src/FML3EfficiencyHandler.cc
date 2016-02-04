#include <fstream>

#include "FastSimulation/ParamL3MuonProducer/interface/FML3EfficiencyHandler.h"
#include "SimDataFormats/Track/interface/SimTrack.h"

#include "Utilities/General/interface/FileInPath.h"
#include "Utilities/General/interface/CMSexception.h"

#include "FastSimulation/Utilities/interface/RandomEngine.h"

FML3EfficiencyHandler::FML3EfficiencyHandler(const RandomEngine * engine)
  : random(engine) {

   std::string fname = "FastSimulation/ParamL3MuonProducer/data/efficiencyL3.data";
   std::string path(getenv("CMSSW_SEARCH_PATH"));
   FileInPath f1(path,fname);
   if ( f1() == 0) {
     std::cout << "File " << fname << " not found in " << path << std::endl;
     throw Genexception(" efficiency list not found for FML3EfficiencyHandler.");
   } else {
     std::cout << "Reading " << f1.name() << std::endl;
   }
   std::ifstream & listfile = *f1();
 
   double eff=0.;
   int nent;
   listfile >> nent;
   if (nent != nEtaBins) { 
     std::cout << " *** ERROR -> FML3EfficiencyHandler : nEta bins " 
	  << nent << " instead of " << nEtaBins << std::endl;
   }
   for (int i=0; i<nEtaBins; i++) {
     listfile >> eff;
     Effic_Eta[i]=eff;
   }
   int iStart=nEtaBins;
   listfile >> nent;
   if (nent != nPhiBins) { 
     std::cout << " *** ERROR -> FML3EfficiencyHandler : nPhi bins "
	       << nent << " instead of " << nPhiBins << std::endl;
   }
   for (int i=iStart; i<iStart+nPhiBins; i++) {
     listfile >> eff;
     Effic_Phi_Barrel[i-iStart]=eff;
   }
   iStart += nPhiBins;
   listfile >> nent;
   if (nent != nPhiBins) { 
     std::cout << " *** ERROR -> FML3EfficiencyHandler : nPhi bins "
	       << nent << " instead of " << nPhiBins << std::endl;
   }
   for (int i=iStart; i<iStart+nPhiBins; i++) {
     listfile >> eff;
     Effic_Phi_Endcap[i-iStart]=eff;
   }
   iStart += nPhiBins;
   listfile >> nent;
   if (nent != nPhiBins) { 
     std::cout << " *** ERROR -> FML3EfficiencyHandler : nPhi bins "
	  << nent << " instead of " << nPhiBins << std::endl;
   }
   for (int i=iStart; i<iStart+nPhiBins; i++) {
     listfile >> eff;
     Effic_Phi_Extern[i-iStart]=eff;
   }

}

FML3EfficiencyHandler::~FML3EfficiencyHandler(){
  delete Effic_Eta;
  delete Effic_Phi_Barrel;
  delete Effic_Phi_Endcap;
  delete Effic_Phi_Extern;
}

bool FML3EfficiencyHandler::kill(const SimTrack & aTrack) {

  
  // At least eight hit in the tracker : To be tuned !!!
  //  if ( aTrack.recHits().size() < 8 ) return false;

  // At least zero reconstructed  Pixel hits : To be tuned !!!
  //  int seed = 0;
  //  for ( unsigned i=1; i<6; ++i ) 
  //    if ( aTrack.isARecHit(i) ) ++seed;
  //  if ( seed < 0 ) return false;

  double myEffEta=0. , myEffPhi=0. , myEffPt=0. , myCorrection=1. , myEff;

  // Eta dependence :
  double eta = fabs(aTrack.momentum().eta());
  if (eta < 2.40) {
    int iEtaBin = (int) ( (eta/2.40) * nEtaBins);
    myEffEta = Effic_Eta[iEtaBin];
  } else return false;

  // Phi and Pt dependence:
  double pt = std::sqrt(aTrack.momentum().perp2());
  double phi = aTrack.momentum().phi();
  if (phi < 0.) {phi = 2* M_PI + phi; }
  int iPhiBin = (int) ((phi/(2*M_PI)) * nPhiBins);
  if (eta<1.04) {
    myEffPhi = Effic_Phi_Barrel[iPhiBin];
    if (pt>40.) myEffPt = 0.9583 - pt*5.82726e-05;
    //    else if (pt>4.10) myEffPt = 0.952*(1.-exp(-(pt-4.072)));
    else if (pt>4.0) myEffPt = 0.952*(1.-sqrt(exp(-(pt-3.3))));
    myCorrection = 1.124;
  }
  else if (eta<2.07) {
    myEffPhi = 1.;  // no visible residual phi structure in the endcaps
    if (pt>173.) myEffPt = 0.991 - pt*3.46562e-05;
    //    else if (pt>3.10) myEffPt = 0.985*(1.-exp(-(pt-3.061)));
    else if (pt>3.0) myEffPt = 0.985*(1.-sqrt(exp(-(pt-2.4))));
    myCorrection = 1.028;
  }
  else if (eta<2.40) {
    myEffPhi = 1.; // no visible residual phi structure in the endcaps
    if (pt>26.) myEffPt = 0.9221 - pt*7.75139e-05;
    else if (pt>3.00) myEffPt = 0.927*(1.-exp(-sqrt(pt-1.617)));
    myCorrection = 1.133;
  }
  else return false;

  /*
  // Special high Pt muons treatment :
  if (pt>400.) {
    double myEffHighPt=1.;
    double pttev = pt/1000.;
    if (eta < 0.3) {
      //      myEffHighPt = 0.952 - 0.033*pttev;
      myEffHighPt = 0.945 - 0.028*pttev;  // fit up to 3.0 TeV
    } else if ( eta < 0.6) {
      //      myEffHighPt = 0.973 - 0.033*pttev;
      myEffHighPt = 0.974 - 0.033*pttev;  // fit up to 3.0 TeV
    } else if ( eta < 0.9) {
      //      myEffHighPt = 0.969 - 0.045*pttev;  
      myEffHighPt = 0.966 - 0.041*pttev;  // fit up to 2.7 TeV
    } else if ( eta < 1.2) {
      //      myEffHighPt = 0.968 - 0.058*pttev;
      myEffHighPt = 0.957 - 0.044*pttev;  // fit up to 2.7 TeV
    } else if ( eta < 1.5) {
      //      myEffHighPt = 0.966 - 0.074*pttev;
      myEffHighPt = 0.944 - 0.045*pttev;  // fit up to 2.4 TeV
    } else if ( eta < 1.8) {
      //      myEffHighPt = 0.955 - 0.11*pttev;
      myEffHighPt = 0.939 - 0.086*pttev;  // fit up to 1.8 TeV
    } else if ( eta < 2.1) {
      //      myEffHighPt = 0.982 - 0.11*pttev;
      myEffHighPt = 0.989 - 0.122*pttev;  // fit up to 1.5 TeV
    } else {
      //     myEffHighPt = 0.958 - 0.14*pttev;
      myEffHighPt = 0.934 - 0.088*pttev;  // fit up to 1.2 TeV
    }
    myEffPt = (myEffPt>myEffHighPt? myEffHighPt : myEffPt);
  }
  */

  myEff = myEffEta*myEffPhi*myEffPt*myCorrection;
  double prob = random->flatShoot();
  return (myEff > prob) ;

}

