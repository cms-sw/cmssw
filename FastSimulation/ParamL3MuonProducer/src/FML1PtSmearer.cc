
#include "FastSimulation/ParamL3MuonProducer/interface/FML1PtSmearer.h"
#include "FastSimDataFormats/L1GlobalMuonTrigger/interface/SimpleL1MuGMTCand.h"

#include "Utilities/General/interface/FileInPath.h"
#include "Utilities/General/interface/CMSexception.h"

#include <cmath>
#include <iostream>
#include <fstream>

#include "FastSimulation/Utilities/interface/RandomEngine.h"

using namespace std;

FML1PtSmearer::FML1PtSmearer(const RandomEngine * engine)
  : random(engine) {

  string fname = "FastSimulation/ParamL3MuonProducer/data/resolutionL1.data";
  //   std::string path(std::getenv("CMSSW_SEARCH__PATH"));
  std::string path(getenv("CMSSW_SEARCH_PATH"));
  FileInPath f1(path,fname);
  if ( f1() == 0) {
    std::cout << "File " << fname << " not found in " << path << std::endl;
    throw Genexception(" resolution list not found for FastMuonLvl1Trigger.");
  } else {
    cout << "Reading " << f1.name() << std::endl;
  }
  std::ifstream & listfile = *f1();
   
  int ind;
  float num=0.;
  for (int ieta=0; ieta<3; ieta++) {
    for (int i=0; i<NPT; i++) {
      float sum = 0.;
      ind = i*NPTL1 + ieta*(NPT*NPTL1);
      for (int j=0; j<NPTL1; j++) {
	listfile >> num;
	sum += num;
	resolution[ind]=sum; 
	ind++;
      }
      ind = i*NPTL1 + ieta*(NPT*NPTL1);
      for (int j=0; j<NPTL1 ; j++){
	resolution[ind] /= sum;
	ind++;
      }
    }
  }
  
}

FML1PtSmearer::~FML1PtSmearer(){}



bool FML1PtSmearer::smear(SimpleL1MuGMTCand * aMuon) {
  // get and smear the pt
  double Pt=aMuon->smearedPt();
  double AbsEta=fabs(aMuon->getMomentum().eta());

  bool StatusCode=true;  
  
  if (AbsEta>2.40 || Pt<3) { StatusCode=false;}

  if (StatusCode){

    int ieta = 0;
    if      (AbsEta<1.04) ieta = 0 ;
    else if (AbsEta<2.07) ieta = 1 ;
    else if (AbsEta<2.40) ieta = 2 ;

    int counter = 0;
    int ipt = IndexOfPtgen(Pt);
    int ind = counter + ipt*NPTL1 + ieta*(NPT*NPTL1);
    double prob = random->flatShoot();
    while ( (prob > resolution[ind]) && counter<NPTL1 ){ 
      counter++;
      ind++;
    }
    counter++;
    aMuon->setPtPacked(counter & 31);
    aMuon->setPtValue(SimpleL1MuGMTCand::ptScale[counter]);

    prob = random->flatShoot();
    if (prob <= ChargeMisIdent(ieta,Pt)) {
      int OldCharge = aMuon->charge(); 
      int NewCharge = -OldCharge;
      aMuon->setCharge(NewCharge);
    } 

  }

  return StatusCode;

}  


int FML1PtSmearer::IndexOfPtgen(float pt) {

  static float vecpt[NPT] = {
                1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,
         10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,
         20.,  21.,  22.,  23.,  24.,  25.,  26.,  27.,  28.,  29.,
         30.,  31.,  32.,  33.,  34.,  35.,  36.,  37.,  38.,  39.,
         40.,  41.,  42.,  43.,  44.,  45.,  46.,  47.,  48.,  49.,
         50.,  51.,  52.,  53.,  54.,  55.,  56.,  57.,  58.,  59.,
         60.,  61.,  62.,  63.,  64.,  65.,  66.,  67.,  68.,  69.,
         70.,  71.,  72.,  73.,  74.,  75.,  76.,  77.,  78.,  79.,
         80.,  81.,  82.,  83.,  84.,  85.,  86.,  87.,  88.,  89.,
         90.,  91.,  92.,  93.,  94.,  95.,  96.,  97.,  98.,  99.,
        100., 110., 120., 130., 140., 150., 160., 170., 180., 190.,
        200., 210., 220., 230., 240., 250., 260., 270., 280., 290.,
        300., 310., 320., 330., 340., 350., 360., 370., 380., 390.,
        400., 500., 600., 700., 800., 900., 1000. };

  for (int i=0; i<NPT; i++) {
    if (pt<vecpt[i]) return i;
  }
  return NPT;
}
