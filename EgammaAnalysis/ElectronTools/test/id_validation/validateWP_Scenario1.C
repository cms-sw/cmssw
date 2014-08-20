#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TCut.h"

TString fname = "electron_ntuple_sc1.root";

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

void validateWP_Scenario1(){


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
      testLocalString += "&&            full5x5_sigmaIetaIeta < 0.012236";
      testLocalString += "&&                      abs(dEtaIn) < 0.021156";
      testLocalString += "&&                      abs(dPhiIn) < 0.247197";
      testLocalString += "&&                           hOverE < 0.241641";
      testLocalString += "&&                  relIsoWithDBeta < 0.239832";
      testLocalString += "&&                          ooEmooP < 0.323747";
      testLocalString += "&&                          abs(d0) < 0.031812";
      testLocalString += "&&                          abs(dz) < 0.499344";
      testLocalString += "&&         expectedMissingInnerHits < 2.000050";
    }else if( iWP == 1){
      // Loose
      testLocalString += "&&            full5x5_sigmaIetaIeta < 0.011836";
      testLocalString += "&&                      abs(dEtaIn) < 0.015792";
      testLocalString += "&&                      abs(dPhiIn) < 0.079938";
      testLocalString += "&&                           hOverE < 0.145705";
      testLocalString += "&&                  relIsoWithDBeta < 0.183981";
      testLocalString += "&&                          ooEmooP < 0.105459";
      testLocalString += "&&                          abs(d0) < 0.019094";
      testLocalString += "&&                          abs(dz) < 0.035802";
      testLocalString += "&&         expectedMissingInnerHits < 1.020000";
    }else if( iWP == 2){
      // Medium      
      testLocalString += "&&            full5x5_sigmaIetaIeta < 0.010326";
      testLocalString += "&&                      abs(dEtaIn) < 0.015382";
      testLocalString += "&&                      abs(dPhiIn) < 0.050766";
      testLocalString += "&&                           hOverE < 0.100644";
      testLocalString += "&&                  relIsoWithDBeta < 0.135355";
      testLocalString += "&&                          ooEmooP < 0.052553";
      testLocalString += "&&                          abs(d0) < 0.012012";
      testLocalString += "&&                          abs(dz) < 0.030398";
      testLocalString += "&&         expectedMissingInnerHits < 1.020000";
    }else if( iWP == 3){
      // Tight
      testLocalString += "&&            full5x5_sigmaIetaIeta < 0.010244";
      testLocalString += "&&                      abs(dEtaIn) < 0.012316";
      testLocalString += "&&                      abs(dPhiIn) < 0.024063";
      testLocalString += "&&                           hOverE < 0.073594";
      testLocalString += "&&                  relIsoWithDBeta < 0.100254";
      testLocalString += "&&                          ooEmooP < 0.025730";
      testLocalString += "&&                          abs(d0) < 0.009088";
      testLocalString += "&&                          abs(dz) < 0.016881";
      testLocalString += "&&         expectedMissingInnerHits < 1.020000";
   } else {
     assert(0);
   }
  }else {
    // endcap
    if( iWP == 0 ){
      // Veto
      testLocalString += "&&            full5x5_sigmaIetaIeta < 0.035176";
      testLocalString += "&&                      abs(dEtaIn) < 0.028286";
      testLocalString += "&&                      abs(dPhiIn) < 0.228815";
      testLocalString += "&&                           hOverE < 0.185910";
      testLocalString += "&&                  relIsoWithDBeta < 0.237643";
      testLocalString += "&&                          ooEmooP < 0.133209";
      testLocalString += "&&                          abs(d0) < 0.216331";
      testLocalString += "&&                          abs(dz) < 0.911467";
      testLocalString += "&&         expectedMissingInnerHits < 3.000050";
    }else if( iWP == 1){
      // Loose      
      testLocalString += "&&            full5x5_sigmaIetaIeta < 0.032479";
      testLocalString += "&&                      abs(dEtaIn) < 0.024830";
      testLocalString += "&&                      abs(dPhiIn) < 0.096950";
      testLocalString += "&&                           hOverE < 0.115660";
      testLocalString += "&&                  relIsoWithDBeta < 0.211836";
      testLocalString += "&&                          ooEmooP < 0.110190";
      testLocalString += "&&                          abs(d0) < 0.098601";
      testLocalString += "&&                          abs(dz) < 0.879581";
      testLocalString += "&&         expectedMissingInnerHits < 1.020000";
    }else if( iWP == 2){
      // Medium            
      testLocalString += "&&            full5x5_sigmaIetaIeta < 0.030011";
      testLocalString += "&&                      abs(dEtaIn) < 0.022962";
      testLocalString += "&&                      abs(dPhiIn) < 0.056415";
      testLocalString += "&&                           hOverE < 0.099253";
      testLocalString += "&&                  relIsoWithDBeta < 0.146530";
      testLocalString += "&&                          ooEmooP < 0.109146";
      testLocalString += "&&                          abs(d0) < 0.067921";
      testLocalString += "&&                          abs(dz) < 0.781003";
      testLocalString += "&&         expectedMissingInnerHits < 1.020000";      
    }else if( iWP == 3){
      // Tight
      testLocalString += "&&            full5x5_sigmaIetaIeta < 0.028612";
      testLocalString += "&&                      abs(dEtaIn) < 0.019221";
      testLocalString += "&&                      abs(dPhiIn) < 0.042714";
      testLocalString += "&&                           hOverE < 0.080394";
      testLocalString += "&&                  relIsoWithDBeta < 0.137026";
      testLocalString += "&&                          ooEmooP < 0.076298";
      testLocalString += "&&                          abs(d0) < 0.037208";
      testLocalString += "&&                          abs(dz) < 0.064947";
      testLocalString += "&&         expectedMissingInnerHits < 1.020000";
    }else{
      assert(0);
    }
  }
  
  return testLocalString;

}

