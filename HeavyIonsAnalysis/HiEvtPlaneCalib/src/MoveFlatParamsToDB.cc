// -*- C++ -*-
//
// Package:    MoveFlatParamsToDB
// Class:      MoveFlatParamsToDB
// 
/**\class MoveFlatParamsToDB MoveFlatParamsToDB.cc HiEvtPlaneFlatten/MoveFlatParamsToDB/src/MoveFlatParamsToDB.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Stephen Sanders
//         Created:  Fri Jun 11 12:56:15 EDT 2010
// $Id: MoveFlatParamsToDB.cc,v 1.2 2011/10/07 09:41:29 yilmaz Exp $
//
//


// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
//#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"

#include "CondFormats/DataRecord/interface/HeavyIonRPRcd.h"
#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneList.h"

#include "CondFormats/HIObjects/interface/RPFlatParams.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2D.h"
#include "TH2F.h"
#include "TTree.h"
#include "TH1I.h"
#include "TF1.h"
#include "TList.h"
#include "TString.h"
#include <time.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <stdio.h>

using namespace std;
using namespace hi;

//
// class declaration
//
static const int MaxEPAllowed = 50;  //determined by RPFlatParams.h

class MoveFlatParamsToDB : public edm::EDAnalyzer {
public:
  explicit MoveFlatParamsToDB(const edm::ParameterSet&);
  ~MoveFlatParamsToDB();
  
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  bool GetRescor(double bin, string epname, double * res, double * eres);
  void AddCentralityBins( double centbin, int nbins);
//   edm::Service<TFileService> fs;
  TFile * inFile;
  //Hard coded limit of 100 different reaction planes
  TH1D * x[NumEPNames];
  TH1D * y[NumEPNames];
  TH1D * xycnt[NumEPNames];
  string rpname[NumEPNames];
  int RPNameIndx[NumEPNames];
  int RPSubEvnt[NumEPNames];
  RPFlatParams * rpFlat;
  int nRP;
  Bool_t rescorInputAvailable;
  double res1[50][100];
  double eres1[50][100];
  double res2[50][50];
  double eres2[50][50];
  double res5[50][20];
  double eres5[50][20];
  double res10[50][10];
  double eres10[50][10];
  double res20[50][5];
  double eres20[50][5];
  double res25[50][4];
  double eres25[50][40];
  double res30[50][3];
  double eres30[50][3];
  double res40[50][2];
  double eres40[50][2];
  // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MoveFlatParamsToDB::MoveFlatParamsToDB(const edm::ParameterSet& iConfig)

{
  rpFlat = new RPFlatParams();
  cout<<"Enter MoveFlatParamsToDB"<<endl;
  //now do what ever initialization is needed
  inFile = new TFile("data/rpflat_combined.root");
  if(inFile->IsZombie()) {
    cout<<"file not found"<<endl;
  }
  TList * list = ((TDirectory *)inFile->Get("hiEvtPlaneFlatCalib"))->GetListOfKeys();
  int indx =0;
  int cnt = 0;
  for(int i = 0; i<NumEPNames; i++) {
    x[i]=0;
    y[i]=0;
    xycnt[i]=0;
  }
  while(indx >=0 && indx<NumEPNames) {
    int EPNameIndx = -1;
    TString name = list->At(indx)->GetName();
    if(!name.Contains("cent")&&!name.Contains("vtx")&&!name.Contains("MidEtaTrackRescor")&&!name.Contains("tree")) {
      for(int i = 0; i<NumEPNames; i++) {
	if(name.CompareTo(EPNames[i])==0) {
	  EPNameIndx = i;
	  break;
	}
      }
      if(EPNameIndx <0) cout<<"A bad reaction plane name has been encountered: "<<name.Data()<<endl;
      RPNameIndx[cnt]=EPNameIndx;
      TString sname = name;
      x[cnt] = (TH1D *) inFile->Get(Form("hiEvtPlaneFlatCalib/%s/x_%s",name.Data(),name.Data()));
      y[cnt] = (TH1D *) inFile->Get(Form("hiEvtPlaneFlatCalib/%s/y_%s",name.Data(),name.Data()));
      xycnt[cnt] = (TH1D *) inFile->Get(Form("hiEvtPlaneFlatCalib/%s/cnt_%s",name.Data(),name.Data()));
      rpname[cnt]=sname;
      if(!x[cnt]) cout<<"bad x"<<endl;
      if(!y[cnt]) cout<<"bad y"<<endl;
      if(!xycnt[cnt]) cout<<"bad cnt"<<endl;
      if(x[cnt] && xycnt[cnt] && y[cnt]) {
	x[cnt]->Divide(xycnt[cnt]);
	y[cnt]->Divide(xycnt[cnt]);
      }
      ++cnt;
      if(cnt>NumEPNames||cnt>MaxEPAllowed) {
	cout<<"Maximum number of reaction planes exceeded!"<<endl;
	break;
      }
    }
    
    if(list->At(indx)==list->Last())
      indx = -1;
      else
	++indx;
  }
  nRP = cnt;
  cout<<"nRP = "<<nRP<<endl;
  FILE * ftest = fopen("RescorTables","r");
  if(ftest) {
    cout<<"Resolution corrections will be added"<<endl;
    rescorInputAvailable = kTRUE;
    fclose(ftest);
  } else {
    cout<<"Resolution corrections not available"<<endl;
    rescorInputAvailable = kFALSE;
  }
}


MoveFlatParamsToDB::~MoveFlatParamsToDB()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//
bool MoveFlatParamsToDB::GetRescor(double bin, string epname, double * restmp, double * erestmp) {
  FILE * ftest;
  ftest = fopen(Form("RescorTables/%s_%04.1f.dat",epname.data(),bin),"r");
  if(!ftest) {
  cout<<Form("RescorTables/%s_%04.1f.dat Not FOUND",epname.data(),bin)<<endl;
    return false; 
  }
  int nbins = 0;
  char buf[80];
  while(fgets(buf,80,ftest)!=NULL) {
    double minc, maxc;
    sscanf(buf,"%lg\t%lg\t%lg\t%lg",&minc, &maxc, &restmp[nbins],&erestmp[nbins]);
    ++nbins;
  }
  return true;
}

void MoveFlatParamsToDB::AddCentralityBins( double centbin, int nbins) {
    RPFlatParams::EP * thisBin = new RPFlatParams::EP();
    for(int i = 0; i<nRP; i++) {
      thisBin->x[i] = centbin;
      thisBin->y[i] = (double) nbins;
      thisBin->RPNameIndx[i] = RPNameIndx[i];
    }
    rpFlat->m_table.push_back(*thisBin);
    for(int j = 0; j<nbins; j++) {
      RPFlatParams::EP * thisBin = new RPFlatParams::EP();
      for(int i = 0; i<nRP; i++) {
	if(fabs(centbin-1.)<0.01) {
	  thisBin->x[i] = res1[i][j];
	  thisBin->y[i] = eres1[i][j];
	}
	if(fabs(centbin-2.)<0.01) {
	  thisBin->x[i] = res2[i][j];
	  thisBin->y[i] = eres2[i][j];
	}
	if(fabs(centbin-5.)<0.01) {
	  thisBin->x[i] = res5[i][j];
	  thisBin->y[i] = eres5[i][j];
	}
	if(fabs(centbin-10.)<0.01) {
	  thisBin->x[i] = res10[i][j];
	  thisBin->y[i] = eres10[i][j];
	}
	if(fabs(centbin-20.)<0.01) {
	  thisBin->x[i] = res20[i][j];
	  thisBin->y[i] = eres20[i][j];
	}
	if(fabs(centbin-25.)<0.01) {
	  thisBin->x[i] = res25[i][j];
	  thisBin->y[i] = eres25[i][j];
	}
	if(fabs(centbin-30.)<0.01) {
	  thisBin->x[i] = res30[i][j];
	  thisBin->y[i] = eres30[i][j];
	}
	if(fabs(centbin-40.)<0.01) {
	  thisBin->x[i] = res40[i][j];
	  thisBin->y[i] = eres40[i][j];
	}
	thisBin->RPNameIndx[i] = RPNameIndx[i];
      } 
      rpFlat->m_table.push_back(*thisBin);
    }
  } 

// ------------ method called to for each event  ------------
void
MoveFlatParamsToDB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  //check which centrality binning is available
  bool cent1 = kFALSE;
  bool cent2 = kFALSE;
  bool cent5 = kFALSE;
  bool cent10 = kFALSE;
  bool cent20 = kFALSE;
  bool cent25 = kFALSE;
  bool cent30 = kFALSE;
  bool cent40 = kFALSE;

  for(int i = 0; i<nRP; i++) {
    if(GetRescor(1.0,EPNames[i],res1[i],eres1[i])) cent1 = kTRUE;
    if(GetRescor(2.0,EPNames[i],res2[i],eres2[i])) cent2 = kTRUE;
    if(GetRescor(5.0,EPNames[i],res5[i],eres5[i])) cent5 = kTRUE;
    if(GetRescor(10.0,EPNames[i],res10[i],eres10[i])) cent10 = kTRUE;
    if(GetRescor(20.0,EPNames[i],res20[i],eres20[i])) cent20 = kTRUE;
    if(GetRescor(25.0,EPNames[i],res25[i],eres25[i])) cent25 = kTRUE;
    if(GetRescor(30.0,EPNames[i],res30[i],eres30[i])) cent30 = kTRUE;
    if(GetRescor(40.0,EPNames[i],res40[i],eres40[i])) cent40 = kTRUE;
  }

  int centreserve = 0;
  if(cent1)  centreserve+=100 + 1;
  if(cent2)  centreserve+=50 + 1;
  if(cent5)  centreserve+=20 + 1;
  if(cent10) centreserve+=10 + 1;
  if(cent20) centreserve+=5 + 1;
  if(cent25) centreserve+=4 + 1;
  if(cent30) centreserve+=3 + 1;
  if(cent40) centreserve+=2 + 1;

  rpFlat->m_table.reserve(x[0]->GetNbinsX() + centreserve);
  for(int j = 0; j<x[0]->GetNbinsX();j++) {
    RPFlatParams::EP * thisBin = new RPFlatParams::EP();
    for(int i = 0; i<nRP; i++) {
      thisBin->x[i] = x[i]->GetBinContent(j+1);
      thisBin->y[i] = y[i]->GetBinContent(j+1);
      thisBin->RPNameIndx[i]=RPNameIndx[i];
    }
    rpFlat->m_table.push_back(*thisBin);
    if(thisBin) delete thisBin;
  }
  if(cent1) AddCentralityBins( 1.0, 100);
  if(cent2) AddCentralityBins( 2.0, 50);
  if(cent5) AddCentralityBins( 5.0, 20);
  if(cent10) AddCentralityBins(10.0, 10);
  if(cent20) AddCentralityBins(20.0, 5);
  if(cent25) AddCentralityBins(25.0, 4);
  if(cent30) AddCentralityBins(30.0, 3);
  if(cent40) AddCentralityBins(40.0, 2);
  edm::Service<cond::service::PoolDBOutputService> pool;
  if(pool.isAvailable()){
    if(pool->isNewTagRequest("HeavyIonRPRcd") ) {
      pool->createNewIOV<RPFlatParams>(rpFlat,pool->beginOfTime(), pool->endOfTime(),"HeavyIonRPRcd");
    } else {
      pool->appendSinceTime<RPFlatParams>(rpFlat, pool->currentTime(),"HeavyIonRPRcd");
    }
  }
}


// ------------ method called once each job just before starting event loop  ------------
void 
MoveFlatParamsToDB::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MoveFlatParamsToDB::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(MoveFlatParamsToDB);
