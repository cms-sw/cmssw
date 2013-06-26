#include "TChain.h"
#include "TClonesArray.h"

void CocoaAna() {

  gSystem->CompileMacro("NtupleObjects.cc","k"); // This file is in CocoaAnalysis/src/

  TChain* chain = new TChain("CocoaTree");
  chain->AddFile("report.root",TChain::kBigNumber);

  double chi2meas;
  double chi2cal;
  int ndof;
  int nfitparam;
  int noptobj;
  int nsensor2d;
  int ndistancemeter;
  int ndistancemeter1dim;
  int ntiltmeter;
  int ncops;

  chain->SetBranchAddress("Chi2Measurements",   &chi2meas);
  chain->SetBranchAddress("Chi2CalibratedParameters",   &chi2cal);
  chain->SetBranchAddress("NDegreesOfFreedom",  &ndof);
  chain->SetBranchAddress("NFitParameters",     &nfitparam);
  chain->SetBranchAddress("NOptObjects",        &noptobj);
  chain->SetBranchAddress("NSensor2D",          &nsensor2d);
  chain->SetBranchAddress("NDistancemeter",     &ndistancemeter);
  chain->SetBranchAddress("NDistancemeter1Dim", &ndistancemeter1dim);
  chain->SetBranchAddress("NTiltmeter",         &ntiltmeter);
  chain->SetBranchAddress("NCops",              &ncops);

  TClonesArray* FitParamA              = new TClonesArray("FitParam");
  TClonesArray* OptObjectA             = new TClonesArray("OptObject");
  TClonesArray* Sensor2DMeasA          = new TClonesArray("Sensor2DMeas");
  TClonesArray* DistancemeterMeasA     = new TClonesArray("DistancemeterMeas");
  TClonesArray* Distancemeter1DimMeasA = new TClonesArray("Distancemeter1DimMeas");
  TClonesArray* TiltmeterMeasA         = new TClonesArray("TiltmeterMeas");
  TClonesArray* CopsMeasA              = new TClonesArray("CopsMeas");

  chain->SetBranchAddress("FitParameters",                  &FitParamA);
  chain->SetBranchAddress("OptObjects",                     &OptObjectA);
  chain->SetBranchAddress("Sensor2DMeasurements",           &Sensor2DMeasA);
  chain->SetBranchAddress("DistancemeterMeasurements",      &DistancemeterMeasA);
  chain->SetBranchAddress("Distancemeter1DimMeasurements",  &Distancemeter1DimMeasA);
  chain->SetBranchAddress("TiltmeterMeasurements",          &TiltmeterMeasA);
  chain->SetBranchAddress("CopsMeasurements",               &CopsMeasA);

  int nEvents = chain -> GetEntries();
  int counter = 0;

  for(int ev=0; ev<nEvents; ev++) {

    counter++;
//     int lflag = chain->LoadTree(ev);
    chain -> GetEntry(ev);
    
//     TBranch *bchi2meas = chain->GetBranch("Chi2Measurements");
//     int nbytes = bchi2meas->GetEntry(lflag);

//     TBranch *bchi2cal = chain->GetBranch("Chi2CalibratedParameters");
//     int nbytes = bchi2cal->GetEntry(lflag);

//     TBranch *bndof = chain->GetBranch("NDegreesOfFreedom");
//     int nbytes = bndof->GetEntry(lflag);


//     TBranch *bnfitparam = chain->GetBranch("NFitParameters");
//     int nbytes = bnfitparam->GetEntry(lflag);

//     TBranch *bfitparam = chain->GetBranch("FitParameters");
//     nbytes = bfitparam->GetEntry(lflag); 
    
//     TBranch *bnoptobj = chain->GetBranch("NOptObjects");
//     int nbytes = bnoptobj->GetEntry(lflag);

//     TBranch *boptobject = chain->GetBranch("OptObjects");
//     nbytes = boptobject->GetEntry(lflag); 
    
//     TBranch *bnsensor2d = chain->GetBranch("NSensor2D");
//     nbytes = bnsensor2d->GetEntry(lflag);

//     TBranch *bsensor2d = chain->GetBranch("Sensor2DMeasurements");
//     nbytes = bsensor2d->GetEntry(lflag); 
    
//     TBranch *bndistancemeter = chain->GetBranch("NDistancemeter");
//     nbytes = bndistancemeter->GetEntry(lflag);

//     TBranch *bdistancemeter = chain->GetBranch("DistancemeterMeasurements");
//     nbytes = bdistancemeter->GetEntry(lflag);  
    
//     TBranch *bndistancemeter1dim = chain->GetBranch("NDistancemeter1Dim");
//     nbytes = bndistancemeter1dim->GetEntry(lflag);

//     TBranch *bdistancemeter1dim = chain->GetBranch("Distancemeter1DimMeasurements");
//     nbytes = bdistancemeter1dim->GetEntry(lflag); 
    
//     TBranch *bntiltmeter = chain->GetBranch("NTiltmeter");
//     nbytes = bntiltmeter->GetEntry(lflag);

//     TBranch *btiltmeter = chain->GetBranch("TiltmeterMeasurements");
//     nbytes = btiltmeter->GetEntry(lflag); 
    
//     TBranch *bncops = chain->GetBranch("NCops");
//     nbytes = bncops->GetEntry(lflag);

//     TBranch *bcops = chain->GetBranch("CopsMeasurements");
//     nbytes = bcops->GetEntry(lflag); 

    cout << chi2meas << " " << chi2cal << " " << ndof << endl;
    
    for (int np = 0; np<nfitparam; np++) {

      FitParam* fitparam = (FitParam*) FitParamA->At(np);
      cout << np << " " << fitparam->Name << " " << fitparam->Quality << " " << " " << fitparam->InitialValue << " " << fitparam->InitialSigma << " " << fitparam->FittedValue << " " << endl;

    }

    for (int no = 0; no<noptobj; no++) {

      OptObject* optobj = (OptObject*) OptObjectA->At(no);
      int pr = optobj->Parent; OptObject* optobjp = (OptObject*) OptObjectA->At(pr);
      cout << "     " << optobj->CentreGlobal[0] << " " << optobj->CentreGlobal[1] << " " << optobj->CentreGlobal[2] << "        " << optobj->CentreLocal[0] << " " << optobj->CentreLocal[1] << " " << optobj->CentreLocal[2] << endl;
      cout << "     " << optobj->AnglesGlobal[0] << " " << optobj->AnglesGlobal[1] << " " << optobj->AnglesGlobal[2] << "        " << optobj->AnglesLocal[0] << " " << optobj->AnglesLocal[1] << " " << optobj->AnglesLocal[2] << endl;
    
    }

    for (int ns = 0; ns<nsensor2d; ns++) {

      Sensor2DMeas* sensor2d = (Sensor2DMeas*) Sensor2DMeasA->At(ns);
      cout << ns << " " << sensor2d->Name << " " << sensor2d->Position[0] << " " << sensor2d->PosError[0] << " " << sensor2d->SimulatedPosition[0] << " " << sensor2d->SimulatedPosition[0]-sensor2d->Position[0] << endl;

    }

    for (int ns = 0; ns<ndistancemeter; ns++) {

      DistancemeterMeas* dist = (DistancemeterMeas*) DistancemeterMeasA->At(ns);
      cout << ns << " " << dist->Name << " " << dist->Distance << " " << dist->DisError << " " << dist->SimulatedDistance << endl;

    }



  }




}
