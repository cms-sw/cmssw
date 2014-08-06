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
#include "DataFormats/Common/interface/EDProduct.h"
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
using namespace std;
using namespace hi;

//
// class declaration
//

class MoveFlatParamsToDB : public edm::EDAnalyzer {
public:
  explicit MoveFlatParamsToDB(const edm::ParameterSet&);
  ~MoveFlatParamsToDB();
  
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
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
    if(!name.Contains("cent")&&!name.Contains("vtx")&&!name.Contains("MidEtaTrackRescor")) {
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
      if(cnt>NumEPNames||cnt>50) {
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
}


MoveFlatParamsToDB::~MoveFlatParamsToDB()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
MoveFlatParamsToDB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  rpFlat = new RPFlatParams();
  rpFlat->m_table.reserve(x[0]->GetNbinsX());
  cout<<"Size of table: "<<x[0]->GetNbinsX()<<endl;
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
  cout<<"Number of RP: "<<nRP<<endl;
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if(poolDbService.isAvailable())
    poolDbService->writeOne( rpFlat,poolDbService->beginOfTime(),"HeavyIonRPRcd");
  cout<<"DONE"<<endl;
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
