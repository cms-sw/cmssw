//   COCOA class implementation file
//Id:  NtupleManager.cc
//CAT: Analysis
//
//   History: v1.0 
//   Luca Scodellaro
#include "TROOT.h"
#include <cstdlib>
#include <fstream>

#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaFit/interface/NtupleManager.h"
#include "Alignment/CocoaFit/interface/FittedEntry.h"
#include "Alignment/CocoaModel/interface/Measurement.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "Alignment/CocoaModel/interface/Entry.h"
// #include "Alignment/CocoaUtilities/interface/ALIUtils.h"
// #include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"
#include "TFile.h" 
#include "TTree.h"
#include "TClonesArray.h"
//#include "TMath.h"

NtupleManager* NtupleManager::instance = 0;

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Gets the only instance of Model
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
NtupleManager* NtupleManager::getInstance()
{
  if(!instance) {
    instance = new NtupleManager;
  }
  return instance;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Book ntuple
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void NtupleManager::BookNtuple()
{
  theRootFile = new TFile("report.root","RECREATE","Simple ROOT Ntuple");

  CocoaTree = new TTree("CocoaTree","CocoaTree"); 

  CocoaTree->Branch("Chi2Measurements",&Chi2Measurements,"Chi2Measurements/D");
  CocoaTree->Branch("Chi2CalibratedParameters",&Chi2CalibratedParameters,"Chi2CalibratedParameters/D");
  CocoaTree->Branch("NDegreesOfFreedom",&NDegreesOfFreedom,"NDegreesOfFreedom/I");

  CloneFitParam = new TClonesArray("FitParam");
  CocoaTree->Branch("FitParameters",&CloneFitParam,32000,2);
  CocoaTree->Branch("NFitParameters",&NFitParameters,"NFitParameters/I");

  CloneOptObject = new TClonesArray("OptObject");
  CocoaTree->Branch("OptObjects",&CloneOptObject,32000,2);
  CocoaTree->Branch("NOptObjects",&NOptObjects,"NOptObjects/I");

  CloneSensor2DMeas = new TClonesArray("Sensor2DMeas");
  CocoaTree->Branch("Sensor2DMeasurements",&CloneSensor2DMeas,32000,2);
  CocoaTree->Branch("NSensor2D",&NSensor2D,"NSensor2D/I");
  
  CloneDistancemeterMeas = new TClonesArray("DistancemeterMeas");
  CocoaTree->Branch("DistancemeterMeasurements",&CloneDistancemeterMeas,32000,2);
  CocoaTree->Branch("NDistancemeter",&NDistancemeter,"NDistancemeter/I");
  
  CloneDistancemeter1DimMeas = new TClonesArray("Distancemeter1DimMeas");
  CocoaTree->Branch("Distancemeter1DimMeasurements",&CloneDistancemeter1DimMeas,32000,2);
  CocoaTree->Branch("NDistancemeter1Dim",&NDistancemeter1Dim,"NDistancemeter1Dim/I");

  CloneTiltmeterMeas = new TClonesArray("TiltmeterMeas");
  CocoaTree->Branch("TiltmeterMeasurements",&CloneTiltmeterMeas,32000,2);
  CocoaTree->Branch("NTiltmeter",&NTiltmeter,"NTiltmeter/I");

  CloneCopsMeas = new TClonesArray("CopsMeas");
  CocoaTree->Branch("CopsMeasurements",&CloneCopsMeas,32000,2);
  CocoaTree->Branch("NCops",&NCops,"NCops/I");

  theRootFile->Add(CocoaTree);

//   FitParametersTree = new TTree("FitParametersTree","FitParametersTree");
//   FitParametersTree->Branch("NFitParameters",&NFitParameters,"NFitParameters/I");
//   BookFitParameters = false;
//   theRootFile->Add(FitParametersTree);

//   MeasurementsTree = new TTree("MeasurementsTree","MeasurementsTree");
//   MeasurementsTree->Branch("NMeasurements",&NMeasurements,"NMeasurements/I");
//   BookMeasurements = false;
//   theRootFile->Add(MeasurementsTree);
 

}
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Init ntuple
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void NtupleManager::InitNtuple()
{
  CloneFitParam->Clear();

  Chi2Measurements = 0.;
  Chi2CalibratedParameters = 0.;
  NDegreesOfFreedom = 0;
  NFitParameters = 0; 
  NOptObjects = 0; 
  NSensor2D = 0; 
  NDistancemeter = 0; 
  NDistancemeter1Dim = 0; 
  NTiltmeter = 0;
  NCops = 0;
}
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fill ntuple tree
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void NtupleManager::FillNtupleTree()
{
  CocoaTree->Fill();
}
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Close ntuple
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void NtupleManager::WriteNtuple()
{
  theRootFile->Write();
  theRootFile->Close();
}
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Close ntuple
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void NtupleManager::FillChi2()
{
  double chi2meas = 0; 
  double chi2cal = 0;
  ALIint nMeas = 0, nUnk = 0;

  //----- Calculate the chi2 of measurements
  std::vector< Measurement* >::const_iterator vmcite;
  for ( vmcite = Model::MeasurementList().begin(); vmcite != Model::MeasurementList().end(); vmcite++) {
    for ( ALIuint ii = 0; ii < ALIuint((*vmcite)->dim()); ii++ ){
      nMeas++;
      double c2 = ( (*vmcite)->value(ii) - (*vmcite)->valueSimulated(ii) ) / (*vmcite)->sigma(ii);
      chi2meas += c2*c2;
    }
  }

  //----- Calculate the chi2 of calibrated parameters
  std::vector< Entry* >::iterator veite;
  for ( veite = Model::EntryList().begin();
	veite != Model::EntryList().end(); veite++ ) {
    if ( (*veite)->quality() == 2 ) nUnk++;
    if ( (*veite)->quality() == 1 ) {
//       std::cout << " " << (*veite)->valueDisplacementByFitting() << " " 
// 		<< (*veite)->value << " " << (*veite)->sigma() << std::endl;
      double c2 = (*veite)->valueDisplacementByFitting() / (*veite)->sigma();
      chi2cal += c2*c2;
    }
  }

  Chi2Measurements = chi2meas;
  Chi2CalibratedParameters = chi2cal;
  NDegreesOfFreedom = nMeas - nUnk;

}
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fill ntuple with fitted parameters
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void NtupleManager::FillFitParameters(MatrixMeschach* AtWAMatrix)
{

//   double ParValue[1000], ParError[1000];
  int theMinEntryQuality = 1; 
  int ii = 0;
  std::vector<Entry*>::const_iterator vecite; 
  for ( vecite = Model::EntryList().begin();
    vecite != Model::EntryList().end(); vecite++ ) {

    //--- Only for good quality parameters (='unk')
    if ( (*vecite)->quality() >= theMinEntryQuality ) {

      ALIint ipos = (*vecite)->fitPos();
      FittedEntry* fe = new FittedEntry( (*vecite), ipos, sqrt(AtWAMatrix->Mat()->me[ipos][ipos]));
//       if (!BookFitParameters) {
// 	CocoaTree->Branch("NFitParameters",&NFitParameters,"NFitParameters/I:");
// 	ALIstring partype = fe->getName() + "/D";
// 	FitParametersTree->Branch(fe->getName().c_str(), &ParValue[ii], partype.c_str());
// 	ALIstring parerrname = fe->getName() + "_err";
// 	ALIstring parerrtype = parerrname + "/D";
// 	FitParametersTree->Branch(parerrname.c_str(), &ParError[ii], parerrtype.c_str());
//       }
//       ParValue[ii] = fe->getValue();
//       ParError[ii] = fe->getSigma();
     std::cout << "EEE " << (*vecite)->ValueDimensionFactor() << " " << (*vecite)->SigmaDimensionFactor() << " " << fe->getOptOName() << " " << fe->getEntryName() << " " << fe->getName() << " " << fe->getOrder() << " " << fe->getQuality() << " " << (*vecite)->type() << " " << std::endl;
      FitParamA = new( (*CloneFitParam)[ii] ) FitParam();
      FitParamA->Name = fe->getName();
      if (fe->getQuality()==1) FitParamA->Quality = "Calibrated";
      else if (fe->getQuality()==2) FitParamA->Quality = "Unknown";
      for (int no = 0; no<NOptObjects; no++) {
	OptObject* optobj = (OptObject*) CloneOptObject->At(no);
	if (optobj->Name==fe->getOptOName()) FitParamA->OptObjectIndex = no;
      }
      float DF = 1.;
      if ((*vecite)->type()=="centre" || (*vecite)->type()=="length") DF = 1000.;
      FitParamA->InitialValue = DF*fe->getOrigValue()*(*vecite)->ValueDimensionFactor();
      FitParamA->InitialSigma = DF*fe->getOrigSigma()*(*vecite)->SigmaDimensionFactor();
      FitParamA->FittedValue  = DF*fe->getValue()*(*vecite)->ValueDimensionFactor();
      FitParamA->FittedSigma  = DF*fe->getSigma()*(*vecite)->SigmaDimensionFactor();
      ii++;

    }

  }
//   BookFitParameters = true;
  NFitParameters = ii;
//   FitParametersTree->Fill();

  /*  
  //---------- Loop sets of entries
  std::vector< FittedEntriesSet* > theFittedEntriesSets;
  std::vector< FittedEntriesSet* >::const_iterator vfescite;
  std::vector< FittedEntry* >::const_iterator vfecite;
  ALIint jj = 1;
  for( vfescite = theFittedEntriesSets.begin(); vfescite != theFittedEntriesSets.end(); vfescite++) {
    //---------- Loop entries
    if( vfescite == theFittedEntriesSets.begin() ) {
    //----- dump entries names if first set 
      ALIint ii = 0;
      for( vfecite = ((*vfescite)->FittedEntries()).begin(); vfecite != ((*vfescite)->FittedEntries()).end(); vfecite++) {
	ALIstring partype = (*vfecite)->getName() + "/D";
	FitParametersTree->Branch((*vfecite)->getName().c_str(), &ParValue[ii], partype.c_str());
	ALIstring parerrname = (*vfecite)->getName() + "_err";
	ALIstring parerrtype = parerrname + "/D";
	FitParametersTree->Branch(parerrname.c_str(), &ParError[ii], parerrtype.c_str());
        ii++;
      }
    }
    ALIint ii = 0;
    for( vfecite = ((*vfescite)->FittedEntries()).begin(); vfecite != ((*vfescite)->FittedEntries()).end(); vfecite++) {
      ParValue[ii] = (*vfecite)->getValue();
      ParError[ii] = (*vfecite)->getSigma();
      ii++;
    }
    NFitParameters = ii;
    FitParametersTree->Fill();
    jj++;
  }
  */

}
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fill ntuple with optical object positions and orientations
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void NtupleManager::FillOptObjects(MatrixMeschach* AtWAMatrix)
{

  int ii = 0;
  std::vector< OpticalObject* >::const_iterator vecobj;
  for ( vecobj = Model::OptOList().begin();
	vecobj != Model::OptOList().end(); vecobj++ ) {
    OptObjectA = new( (*CloneOptObject)[ii] ) OptObject();

    OptObjectA->Name =  (*vecobj)->name();
    OptObjectA->Type = (*vecobj)->type();

    if (!(*vecobj)->parent()) {
      OptObjectA->Parent = ii;
      ii++;
      continue;
    }

    int pp = 0;
    std::vector< OpticalObject* >::const_iterator vecobj2;
    for ( vecobj2 = Model::OptOList().begin();
	  vecobj2 != Model::OptOList().end(); vecobj2++ ) {
      if ((*vecobj2)->name()==(*vecobj)->parent()->name()) {
	OptObjectA->Parent = pp;
	continue;
      }
      pp++;
    }
    
    OptObjectA->CentreGlobal[0] = 1000.*(*vecobj)->centreGlobal().x();
    OptObjectA->CentreGlobal[1] = 1000.*(*vecobj)->centreGlobal().y();
    OptObjectA->CentreGlobal[2] = 1000.*(*vecobj)->centreGlobal().z();

    OptObjectA->CentreLocal[0] = 1000.*(*vecobj)->centreLocal().x();
    OptObjectA->CentreLocal[1] = 1000.*(*vecobj)->centreLocal().y();
    OptObjectA->CentreLocal[2] = 1000.*(*vecobj)->centreLocal().z();

    OptObjectA->AnglesLocal[0] = (*vecobj)->getEntryRMangle(XCoor);
    OptObjectA->AnglesLocal[1] = (*vecobj)->getEntryRMangle(YCoor);
    OptObjectA->AnglesLocal[2] = (*vecobj)->getEntryRMangle(ZCoor);

    double theta[3];
    GetGlobalAngles((*vecobj)->rmGlob(), theta);
    for (int i = 0; i<3; i++) OptObjectA->AnglesGlobal[i] = theta[i];

    ii++;

  }

  NOptObjects = ii;


}
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fill ntuple with measurements
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void NtupleManager::FillMeasurements()
{
  //---------- Loop Measurements
  int ss = 0, dd = 0, d1 = 0, tt = 0, cc = 0;
  std::vector< Measurement* >::const_iterator vmcite;
  for ( vmcite = Model::MeasurementList().begin(); vmcite != Model::MeasurementList().end(); vmcite++) {
    std::vector<ALIstring> optonamelist = (*vmcite)->OptONameList(); 
    int last = optonamelist.size() - 1; ALIstring LastOptOName = optonamelist[last];
    int optoind = -999;
    for (int no = 0; no<NOptObjects; no++) {
      OptObject* optobj = (OptObject*) CloneOptObject->At(no);
      if (optobj->Name==LastOptOName) optoind = no;
    }
    //std::cout << "DimSens " << (*vmcite)->type() << " " << (*vmcite)->sigma(0) << " " << LastOptOName << " " << optoind << std::endl;
    if ((*vmcite)->type()=="SENSOR2D") {
      Sensor2DMeasA = new( (*CloneSensor2DMeas)[ss] ) Sensor2DMeas();
      Sensor2DMeasA->Name = (*vmcite)->name();
      Sensor2DMeasA->OptObjectIndex = optoind;
      for (ALIuint i = 0; i<(*vmcite)->dim(); i++) {
	Sensor2DMeasA->Position[i] = 1000.*(*vmcite)->value()[i];
	Sensor2DMeasA->PosError[i] = 1000.*(*vmcite)->sigma()[i];
	Sensor2DMeasA->SimulatedPosition[i] = 1000.*(*vmcite)->valueSimulated(i);
      }
      ss++;
    }
    if ((*vmcite)->type()=="DISTANCEMETER") {
      DistancemeterMeasA = new( (*CloneDistancemeterMeas)[dd] ) DistancemeterMeas();
      DistancemeterMeasA->Name = (*vmcite)->name();
      DistancemeterMeasA->OptObjectIndex = optoind;
      DistancemeterMeasA->Distance = 1000.*(*vmcite)->value()[0];
      DistancemeterMeasA->DisError = 1000.*(*vmcite)->sigma()[0];
      DistancemeterMeasA->SimulatedDistance = 1000.*(*vmcite)->valueSimulated(0);
      dd++;
    }
    if ((*vmcite)->type()=="DISTANCEMETER1DIM") {
      Distancemeter1DimMeasA = new( (*CloneDistancemeter1DimMeas)[d1] ) Distancemeter1DimMeas();
      Distancemeter1DimMeasA->Name = (*vmcite)->name();
      Distancemeter1DimMeasA->OptObjectIndex = optoind;
      Distancemeter1DimMeasA->Distance = 1000.*(*vmcite)->value()[0];
      Distancemeter1DimMeasA->DisError = 1000.*(*vmcite)->sigma()[0];
      Distancemeter1DimMeasA->SimulatedDistance = 1000.*(*vmcite)->valueSimulated(0);
      d1++;

    }
    if ((*vmcite)->type()=="TILTMETER") {
      TiltmeterMeasA = new( (*CloneTiltmeterMeas)[tt] ) TiltmeterMeas();
      TiltmeterMeasA->Name = (*vmcite)->name();
      TiltmeterMeasA->OptObjectIndex = optoind;
      TiltmeterMeasA->Angle    = (*vmcite)->value()[0];
      TiltmeterMeasA->AngError = (*vmcite)->sigma()[0];
      TiltmeterMeasA->SimulatedAngle = (*vmcite)->valueSimulated(0);
      tt++;
    }
    if ((*vmcite)->type()=="COPS") {
      CopsMeasA = new( (*CloneCopsMeas)[cc] ) CopsMeas();
      CopsMeasA->Name = (*vmcite)->name();
      CopsMeasA->OptObjectIndex = optoind;
      for (ALIuint i = 0; i<(*vmcite)->dim(); i++) {
	CopsMeasA->Position[i] = 1000.*(*vmcite)->value()[i];
	CopsMeasA->PosError[i] = 1000.*(*vmcite)->sigma()[i];
	CopsMeasA->SimulatedPosition[i] = 1000.*(*vmcite)->valueSimulated(i);
      }
      cc++;
    }
  }
  NSensor2D = ss; 
  NDistancemeter = dd; 
  NDistancemeter1Dim = d1; 
  NTiltmeter = tt; 
  NCops = cc; 
  //   MeasurementsTree->Fill();
}
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Get global angles from global matrix rotation
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void NtupleManager::GetGlobalAngles(const CLHEP::HepRotation& rmGlob, double *theta)
{

  double xx = rmGlob.xx(); if (fabs(xx)<1.e-08) xx = 0.;
  double xy = rmGlob.xy(); if (fabs(xy)<1.e-08) xy = 0.;
  double xz = rmGlob.xz(); if (fabs(xz)<1.e-08) xz = 0.;
  double yx = rmGlob.yx(); if (fabs(yx)<1.e-08) yx = 0.;
  double yy = rmGlob.yy(); if (fabs(yy)<1.e-08) yy = 0.;
  double yz = rmGlob.yz(); if (fabs(yz)<1.e-08) yz = 0.;
  double zx = rmGlob.zx(); if (fabs(zx)<1.e-08) zx = 0.;
  double zy = rmGlob.zy(); if (fabs(zy)<1.e-08) zy = 0.;
  double zz = rmGlob.zz(); if (fabs(zz)<1.e-08) zz = 0.;

  double beta = asin(-zx);

  double alpha, gamma;
  if (fabs(zx)!=1.) {
  
    double sinalpha = zy/cos(beta);
    double cosalpha = zz/cos(beta);
    if (cosalpha>=0) alpha = asin(sinalpha);
  else alpha = M_PI - asin(sinalpha);
  if (alpha>M_PI) alpha -= 2*M_PI;
  
    double singamma = yx/cos(beta);
    double cosgamma = xx/cos(beta);
    if (cosgamma>=0) gamma = asin(singamma);
    else gamma = M_PI - asin(singamma);
    if (gamma>M_PI) gamma -= 2*M_PI;
    
  } else {

    alpha = 0.;
    
    double singamma = yz/sin(beta);
    double cosgamma = yy;
    if (cosgamma>=0) gamma = asin(singamma);
    else gamma = M_PI - asin(singamma);
    if (gamma>M_PI) gamma -= 2*M_PI;

  }

  int GotGlobalAngles = 0;
  if (fabs(xy-(sin(alpha)*sin(beta)*cos(gamma)-sin(gamma)*cos(alpha)))>1.e-08)
    GotGlobalAngles += 1;
  if (fabs(xz-(cos(alpha)*sin(beta)*cos(gamma)+sin(gamma)*sin(alpha)))>1.e-08) 
    GotGlobalAngles += 10;
  if (fabs(yy-(sin(alpha)*sin(beta)*sin(gamma)+cos(gamma)*cos(alpha)))>1.e-08)
    GotGlobalAngles += 100;
  if (fabs(yz-(cos(alpha)*sin(beta)*sin(gamma)-cos(gamma)*sin(alpha)))>1.e-08) 
    GotGlobalAngles += 1000;
  if (GotGlobalAngles>0) 
    std::cout << "NtupleManager Warning: cannot get global rotation: " 
	      << GotGlobalAngles << std::endl;

  theta[0] = alpha;
  theta[1] = beta;
  theta[2] = gamma;

}

