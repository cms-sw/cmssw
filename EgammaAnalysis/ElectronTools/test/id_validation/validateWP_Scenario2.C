#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TCut.h"

TString fname = "electron_ntuple_sc2.root";

const int nWP = 4;
TString treename[nWP] = {
  "wp1/ElectronTree",
  "wp2/ElectronTree",
  "wp3/ElectronTree",
  "wp4/ElectronTree"
};
TString wpName[4] = {
  "WP Veto",
  "WP Loose",
  "WP Medium",
  "WP Tight"
};

const TCut ptCut = "pt>=20 && pt<50";
const TCut etaCutBarrel = "abs(etaSC)<1.4442";
const TCut etaCutEndcap = "abs(etaSC)>1.566 && abs(etaSC)<2.5";
const TCut otherCuts = "isTrue==1 && abs(dz)<1";

TString getCutString(int iWP, bool isBarrel);

void processWP(int iWP, bool isBarrel, TTree *tree);

void validateWP_Scenario2(){


  TFile *fin = new TFile(fname);
  if( !fin) 
    assert(0);

  for(int i=0; i<nWP; i++){
    TTree *tree = (TTree*)fin->Get(treename[i]);
    if( !tree )
      assert(0);
    printf("\nValidate %s:\n", wpName[i].Data());
    processWP(i, true, tree);
    processWP(i, false, tree);
  }

};

void processWP(int iWP, bool isBarrel, TTree *tree){

  TCut etaCut = etaCutBarrel;
  if( !isBarrel)
    etaCut = etaCutEndcap;

  TCut preselection = ptCut && etaCut && otherCuts;

  TCut testVid = " isPass == 1 ";

  float effVid = (1.0*tree->GetEntries(testVid && preselection) )
    / tree->GetEntries( preselection );

  TCut testLocal( getCutString(iWP, isBarrel));

  float effLocal = (1.0*tree->GetEntries(testLocal && preselection) )
    / tree->GetEntries( preselection );

  double reldiff = fabs(effVid-effLocal)/effLocal;
  TString result = "GOOD MATCH";
  if( reldiff >1e-6 )
    result = "PROBLEMS";

  if( isBarrel )
    printf("   barrel     ");
  else
    printf("   endcap     ");

  printf("VID eff= %.2f    Local calc eff= %.2f  %s\n", 100*effVid, 100*effLocal, result.Data());  

}

TString getCutString(int iWP, bool isBarrel){

  TString testLocalString = "1 ";
  testLocalString += "&& passConversionVeto == 1";

  if( isBarrel ){
    // barrel working points
   if( iWP == 0 ){
      // Veto
      testLocalString += "&&            full5x5_sigmaIetaIeta < 0.0125";
      testLocalString += "&&                      abs(dEtaIn) < 0.02";
      testLocalString += "&&                      abs(dPhiIn) < 0.2579";
      testLocalString += "&&                           hOverE < 0.2564";
      testLocalString += "&&                  relIsoWithDBeta < 0.3313";
      testLocalString += "&&                          ooEmooP < 0.1508";
      testLocalString += "&&                          abs(d0) < 0.025";
      testLocalString += "&&                          abs(dz) < 0.5863";
      testLocalString += "&&         expectedMissingInnerHits < 2.000050";
    }else if( iWP == 1){
      // Loose
      testLocalString += "&&            full5x5_sigmaIetaIeta < 0.0123";
      testLocalString += "&&                      abs(dEtaIn) < 0.0181";
      testLocalString += "&&                      abs(dPhiIn) < 0.0936";
      testLocalString += "&&                           hOverE < 0.141";
      testLocalString += "&&                  relIsoWithDBeta < 0.24";
      testLocalString += "&&                          ooEmooP < 0.1353";
      testLocalString += "&&                          abs(d0) < 0.0166";
      testLocalString += "&&                          abs(dz) < 0.54342";
      testLocalString += "&&         expectedMissingInnerHits < 1.020000";
    }else if( iWP == 2){
      // Medium      
      testLocalString += "&&            full5x5_sigmaIetaIeta < 0.0107";
      testLocalString += "&&                      abs(dEtaIn) < 0.0106";
      testLocalString += "&&                      abs(dPhiIn) < 0.0323";
      testLocalString += "&&                           hOverE < 0.067";
      testLocalString += "&&                  relIsoWithDBeta < 0.218"; // rounded from 0.2179
      testLocalString += "&&                          ooEmooP < 0.104"; // rounded from 0.1043
      testLocalString += "&&                          abs(d0) < 0.0131";
      testLocalString += "&&                          abs(dz) < 0.223"; // rounded from 0.22310
      testLocalString += "&&         expectedMissingInnerHits < 1.020000";
    }else if( iWP == 3){
      // Tight
      testLocalString += "&&            full5x5_sigmaIetaIeta < 0.0106";
      testLocalString += "&&                      abs(dEtaIn) < 0.0091";
      testLocalString += "&&                      abs(dPhiIn) < 0.031";
      testLocalString += "&&                           hOverE < 0.0532";
      testLocalString += "&&                  relIsoWithDBeta < 0.1649";
      testLocalString += "&&                          ooEmooP < 0.0609";
      testLocalString += "&&                          abs(d0) < 0.0126";
      testLocalString += "&&                          abs(dz) < 0.0116";
      testLocalString += "&&         expectedMissingInnerHits < 1.020000";
   } else {
     assert(0);
   }
  }else {
    // endcap
    if( iWP == 0 ){
      // Veto
      testLocalString += "&&            full5x5_sigmaIetaIeta < 0.0371";
      testLocalString += "&&                      abs(dEtaIn) < 0.0141";
      testLocalString += "&&                      abs(dPhiIn) < 0.2591";
      testLocalString += "&&                           hOverE < 0.1335";
      testLocalString += "&&                  relIsoWithDBeta < 0.3816";
      testLocalString += "&&                          ooEmooP < 0.1542";
      testLocalString += "&&                          abs(d0) < 0.2232";
      testLocalString += "&&                          abs(dz) < 0.9513";
      testLocalString += "&&         expectedMissingInnerHits < 3.000050";
    }else if( iWP == 1){
      // Loose      
      testLocalString += "&&            full5x5_sigmaIetaIeta < 0.035";
      testLocalString += "&&                      abs(dEtaIn) < 0.0124";
      testLocalString += "&&                      abs(dPhiIn) < 0.0642";
      testLocalString += "&&                           hOverE < 0.112"; // rounded from 0.1115
      testLocalString += "&&                  relIsoWithDBeta < 0.3529";
      testLocalString += "&&                          ooEmooP < 0.1443";
      testLocalString += "&&                          abs(d0) < 0.098";
      testLocalString += "&&                          abs(dz) < 0.919"; // rounded from 0.9187
      testLocalString += "&&         expectedMissingInnerHits < 1.020000";
    }else if( iWP == 2){
      // Medium            
      testLocalString += "&&            full5x5_sigmaIetaIeta < 0.0318";
      testLocalString += "&&                      abs(dEtaIn) < 0.0108";
      testLocalString += "&&                      abs(dPhiIn) < 0.0455";
      testLocalString += "&&                           hOverE < 0.097";
      testLocalString += "&&                  relIsoWithDBeta < 0.254";
      testLocalString += "&&                          ooEmooP < 0.1201";
      testLocalString += "&&                          abs(d0) < 0.0845";
      testLocalString += "&&                          abs(dz) < 0.7523";
      testLocalString += "&&         expectedMissingInnerHits < 1.020000";      
    }else if( iWP == 3){
      // Tight
      testLocalString += "&&            full5x5_sigmaIetaIeta < 0.0305";
      testLocalString += "&&                      abs(dEtaIn) < 0.0106";
      testLocalString += "&&                      abs(dPhiIn) < 0.0359";
      testLocalString += "&&                           hOverE < 0.0835";
      testLocalString += "&&                  relIsoWithDBeta < 0.2075";
      testLocalString += "&&                          ooEmooP < 0.1126";
      testLocalString += "&&                          abs(d0) < 0.0163";
      testLocalString += "&&                          abs(dz) < 0.5999";
      testLocalString += "&&         expectedMissingInnerHits < 1.020000";
    }else{
      assert(0);
    }
  }
  
  return testLocalString;

}

