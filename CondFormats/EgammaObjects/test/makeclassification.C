#if !defined(__CINT__) || defined(__MAKECINT__)
#include "TMVA/Reader.h"
#include "TMVA/MethodBDT.h"
#include "TFile.h"
#include "GBRForest.h"
#include "Cintex/Cintex.h"
#endif

void makeclassification() {
  
  Float_t *vars = new Float_t[10];
  
  //initialize TMVA Reader (example here is diphoton mva from higgs->gamma gamma mva analysis)
  TMVA::Reader* tmva = new TMVA::Reader();
  tmva->AddVariable("masserrsmeared/mass",            &vars[0]);
  tmva->AddVariable("masserrsmearedwrongvtx/mass",    &vars[1]);
  tmva->AddVariable("vtxprob",                        &vars[2]);
  tmva->AddVariable("ph1.pt/mass",                    &vars[3]);
  tmva->AddVariable("ph2.pt/mass",                    &vars[4]);
  tmva->AddVariable("ph1.eta",                        &vars[5]);
  tmva->AddVariable("ph2.eta",                        &vars[6]);
  tmva->AddVariable("TMath::Cos(ph1.phi-ph2.phi)"   , &vars[7]);
  tmva->AddVariable("ph1.idmva",                      &vars[8]);
  tmva->AddVariable("ph2.idmva",                      &vars[9]);
  
  tmva->BookMVA("BDTG","/afs/cern.ch/user/b/bendavid/cmspublic/diphotonmvaApr1/HggBambu_SMDipho_Jan16_BDTG.weights.xml");
  //tmva->BookMVA("BDTG","/scratch/bendavid/root/HggBambu_SMDipho_Jan16_BDTG.weights.xml");
  
  TMVA::MethodBDT *bdt = dynamic_cast<TMVA::MethodBDT*>(tmva->FindMVA("BDTG"));

 
  //enable root i/o for objects with reflex dictionaries in standalone root mode
  ROOT::Cintex::Cintex::Enable();   

  
  //open output root file
  TFile *fout = new TFile("gbrtest.root","RECREATE");
  
  //create GBRForest from tmva object
  GBRForest *gbr = new GBRForest(bdt);  
  
  //write to file
  fout->WriteObject(gbr,"gbrtest");

  fout->Close();
  
  
}