#include <fstream>

#include "FastSimulation/ParamL3MuonProducer/interface/FMGLfromL3EfficiencyHandler.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
 
#include "Utilities/General/interface/FileInPath.h"
#include "Utilities/General/interface/CMSexception.h"
 
#include "FastSimulation/Utilities/interface/RandomEngine.h"

FMGLfromL3EfficiencyHandler::FMGLfromL3EfficiencyHandler(const RandomEngine * engine)
  : random(engine) {

  std::string fname = "FastSimulation/ParamL3MuonProducer/data/efficiencyGL_L3.data";
  std::string path(getenv("CMSSW_SEARCH_PATH"));
  FileInPath f1(path,fname);
 
  if ( f1() == 0) {
    std::cout << "File " << fname << " not found in " << path << std::endl;
    throw Genexception(" efficiency list not found for FMGLfromL3EfficiencyHandler.");
  } else {
    std::cout << "Reading " << f1.name() << std::endl;
  }
  std::ifstream & listfile = *f1();
  
  double eff=0.;
  int nent;
  listfile >> nent;
  if (nent != nEtaBins) { 
    std::cout << " *** ERROR -> FMGLfromL3EfficiencyHandler : nEta bins " 
	      << nent << " instead of " << nEtaBins << std::endl;
  }
  for (int i=0; i<nEtaBins; i++) {
    listfile >> eff;
    Effic_Eta[i]=eff;
  }

}

FMGLfromL3EfficiencyHandler::~FMGLfromL3EfficiencyHandler(){
  delete Effic_Eta;
}

bool FMGLfromL3EfficiencyHandler::kill(const SimTrack & aTrack) {

  // At least eight hit in the tracker : To be tuned !!!
  //if ( aTrack.recHits().size() < 8 ) return false;

  // At least zero reconstructed  Pixel hits : To be tuned !!!
  //int seed = 0;
  //for ( unsigned i=1; i<6; ++i ) 
  //  if ( aTrack.isARecHit(i) ) ++seed;
  //if ( seed < 0 ) return false;

  double myEffEta=0. , myCorrection=1. , myEff;

  // Eta dependence : 
  double eta = fabs(aTrack.momentum().eta());
  if (eta < 2.40) {
    int iEtaBin = (int) ( (eta/2.40) * nEtaBins);
    myEffEta = Effic_Eta[iEtaBin];
  } else return false;

  /*
  double pt = aTrack.momentum().pt();
  if (eta<1.04) {
    if (pt>40.) myEffPt = 0.9583 - pt*5.82726e-05;
    else if (pt>4.10) myEffPt = 0.952*(1.-exp(-(pt-4.072)));
    myCorrection = 1.115;
  }
  else if (eta<2.07) {
    if (pt>173.) myEffPt = 0.991 - pt*3.46562e-05;
    else if (pt>3.10) myEffPt = 0.985*(1.-exp(-(pt-3.061)));
    myCorrection = 1.034;
  }
  else if (eta<2.40) {
    if (pt>26.) myEffPt = 0.9221 - pt*7.75139e-05;
    else if (pt>3.00) myEffPt = 0.927*(1.-exp(-sqrt(pt-1.617)));
    myCorrection = 1.157;
  }
  else return false;
  */

  //  myEff = myEffEta*myEffPt*myCorrection;
  myEff = myEffEta;
  double prob = random->flatShoot();
  return (myEff > prob) ;

}
