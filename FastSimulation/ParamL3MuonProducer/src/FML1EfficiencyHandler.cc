#include <fstream>

#include "FastSimulation/ParamL3MuonProducer/interface/FML1EfficiencyHandler.h"
#include "FastSimulation/ParamL3MuonProducer/interface/SimpleL1MuGMTCand.h"
#include <FastSimulation/Event/interface/FSimTrack.h>

#include "Utilities/General/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/General/interface/CMSexception.h"

#include "FastSimulation/Utilities/interface/RandomEngine.h"

using namespace std;

FML1EfficiencyHandler::FML1EfficiencyHandler(const RandomEngine * engine)
  : random(engine){

   string fname = "FastSimulation/ParamL3MuonProducer/data/efficiencyL1.data";
   //   std::string path(std::getenv("CMSSW_SEARCH__PATH"));
   std::string path(getenv("CMSSW_SEARCH_PATH"));
   FileInPath f1(path,fname);
   if ( f1() == 0) {
     std::cout << "File " << fname << " not found in " << path << std::endl;
     throw Genexception(" efficiency list not found for FML1EfficiencyHandler.");
   } else {
     std::cout << "Reading " << f1.name() << std::endl;
   }
   std::ifstream & listfile = *f1();

   double eff=0.;
   for (int i=0; i<nEtaBins; i++) {
     listfile >> eff;
     Effic_Eta[i]=eff;
   }
   int iStart=nEtaBins;
   for (int i=iStart; i<iStart+nPhiBins; i++) {
     listfile >> eff;
     Effic_Phi_Barrel[i-iStart]=eff;
   }
   iStart += nPhiBins;
   for (int i=iStart; i<iStart+nPhiBins; i++) {
     listfile >> eff;
     Effic_Phi_Endcap[i-iStart]=eff;
   }
   iStart += nPhiBins;
   for (int i=iStart; i<iStart+nPhiBins; i++) {
     listfile >> eff;
     Effic_Phi_Extern[i-iStart]=eff;
   }

}

FML1EfficiencyHandler::~FML1EfficiencyHandler() {}


bool FML1EfficiencyHandler::kill(const SimpleL1MuGMTCand * aMuon) {

  double myEffEta=0. , myEffPhi=0. , myEff;
  double AbsEta = fabs(aMuon->getMomentum().eta());
  double Phi = aMuon->getMomentum().phi();
  double Pt = std::sqrt(aMuon->getMomentum().perp2());

// efficiency as a function of |eta|
  
  if (AbsEta < 2.40) {
    int iEtaBin = (int) ( (AbsEta/2.40) * nEtaBins);
    myEffEta = Effic_Eta[iEtaBin];
  } else { myEffEta = 0.0; }

// efficiency as a function of phi and combined efficiency:
  
  if (Phi < 0.) {Phi = 2* M_PI + Phi; }
  int iPhiBin = (int) ((Phi/(2*M_PI)) * nPhiBins);

  int ieta = 0;
  if (AbsEta < 1.04) {
    myEffPhi = Effic_Phi_Barrel[iPhiBin];
    myEff = myEffEta * myEffPhi * tuningfactor(0);
    ieta = 0;
  } else if (AbsEta < 2.07) {
    myEffPhi = Effic_Phi_Endcap[iPhiBin];
    myEff = myEffEta * myEffPhi * tuningfactor(1);
    ieta = 1;
  } else if (AbsEta < 2.40) {
    myEffPhi = Effic_Phi_Extern[iPhiBin];
    myEff = myEffEta * myEffPhi * tuningfactor(2);
    ieta = 2;
 } else { myEff = 0. ; }

// Drop of efficiency at the lowest Pt's:
  if (Pt<6) myEff *= dumpingfactor(ieta,Pt);  
   
  double prob =  random->flatShoot();

  return (myEff > prob);
}
