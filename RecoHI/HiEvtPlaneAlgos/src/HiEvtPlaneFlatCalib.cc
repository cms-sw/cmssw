// -*- C++ -*-
//
// Package:    HiEvtPlaneFlatCalib
// Class:      HiEvtPlaneFlatCalib
// 
/**\class HiEvtPlaneFlatCalib HiEvtPlaneFlatCalib.cc HiEvtPlaneFlatten/HiEvtPlaneFlatCalib/src/HiEvtPlaneFlatCalib.cc


 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Stephen Sanders
//         Created:  Sat Jun 26 16:04:04 EDT 2010
// $Id: HiEvtPlaneFlatCalib.cc,v 1.5 2011/11/06 23:17:46 ssanders Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Math/Vector3D.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"

#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/DataRecord/interface/HeavyIonRPRcd.h"
#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/HIObjects/interface/RPFlatParams.h"

#include "RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneFlatten.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2D.h"
#include "TH2F.h"
#include "TTree.h"
#include "TH1I.h"
#include "TF1.h"
#include "TList.h"
#include "TString.h"
//#include "TRandom3.h"
#include <time.h>
#include <cstdlib>
#include <vector>

#include "RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneList.h"

using namespace std;
using namespace hi;

//
// class declaration
//

static const  int NumCentBins=14;
static double wcent[]={0,5,10,15,20,25,30,35,40,50,60,70,80,90,100};

class HiEvtPlaneFlatCalib : public edm::EDAnalyzer {
   public:
      explicit HiEvtPlaneFlatCalib(const edm::ParameterSet&);
      ~HiEvtPlaneFlatCalib();

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
  edm::Service<TFileService> fs;
  edm::InputTag vtxCollection_;
  edm::InputTag inputPlanes_;
  edm::InputTag centrality_;

  int vs_sell;   // vertex collection size
  float vzr_sell;
  float vzErr_sell;
  bool FirstEvent;

  TH1D * hcent;
  TH1D * hvtx;
  TH1D * flatXhist[NumEPNames];
  TH1D * flatYhist[NumEPNames];
  TH1D * flatCnthist[NumEPNames];

  TH1D * flatXDBhist[NumEPNames];
  TH1D * flatYDBhist[NumEPNames];
  TH1D * hPsi[NumEPNames];
  TH1D * hPsiFlat[NumEPNames];
  TH1D * hPsiFlatCent[NumEPNames][NumCentBins];
  TH1D * hPsiFlatSub1[NumEPNames];
  TH1D * hPsiFlatSub2[NumEPNames];
  Double_t epang[NumEPNames];
  HiEvtPlaneFlatten * flat[NumEPNames];
  RPFlatParams * rpFlat;
  int nRP;
  bool genFlatPsi_;
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
HiEvtPlaneFlatCalib::HiEvtPlaneFlatCalib(const edm::ParameterSet& iConfig)
{
  vtxCollection_  = iConfig.getParameter<edm::InputTag>("vtxCollection_");
  inputPlanes_ = iConfig.getParameter<edm::InputTag>("inputPlanes_");
  centrality_ = iConfig.getParameter<edm::InputTag>("centrality_");
  genFlatPsi_ = iConfig.getUntrackedParameter<bool>("genFlatPsi_",true);
  FirstEvent = kTRUE;
  //now do what ever other initialization is needed
  hcent = fs->make<TH1D>("cent","cent",41,0,40);
  hvtx = fs->make<TH1D>("vtx","vtx",1000,-50,50);
  Int_t FlatOrder = 21;
  for(int i = 0; i<NumEPNames; i++) {
    TFileDirectory subdir = fs->mkdir(Form("%s",EPNames[i].data()));
    flat[i] = new HiEvtPlaneFlatten();
    flat[i]->Init(FlatOrder,20,2,EPNames[i],EPOrder[i]);
    int nbins = flat[i]->GetHBins();
    flatXhist[i] = subdir.make<TH1D>(Form("x_%s",EPNames[i].data()),Form("x_%s",EPNames[i].data()),nbins,-0.5,nbins-0.5);
    flatYhist[i] = subdir.make<TH1D>(Form("y_%s",EPNames[i].data()),Form("y_%s",EPNames[i].data()),nbins,-0.5,nbins-0.5);
    flatXDBhist[i] = subdir.make<TH1D>(Form("xDB_%s",EPNames[i].data()),Form("x_%s",EPNames[i].data()),nbins,-0.5,nbins-0.5);
    flatYDBhist[i] = subdir.make<TH1D>(Form("yDB_%s",EPNames[i].data()),Form("y_%s",EPNames[i].data()),nbins,-0.5,nbins-0.5);
    flatCnthist[i] = subdir.make<TH1D>(Form("cnt_%s",EPNames[i].data()),Form("cnt_%s",EPNames[i].data()),nbins,-0.5,nbins-0.5);
    Double_t psirange = 4;
    if(EPOrder[i]==2 ) psirange = 2;
    if(EPOrder[i]==3 ) psirange = 1.5;
    if(EPOrder[i]==4 ) psirange = 1;
    if(EPOrder[i]==5) psirange = 0.8;
    if(EPOrder[i]==6) psirange = 0.6;
    hPsi[i] = subdir.make<TH1D>("psi","psi",800,-psirange,psirange);
    hPsi[i]->SetXTitle("#Psi");
    hPsi[i]->SetYTitle(Form("Counts (cent<80%c)",'%'));
    hPsiFlat[i] = subdir.make<TH1D>("psiFlat","psiFlat",800,-psirange,psirange);
    hPsiFlat[i]->SetXTitle("#Psi");
    hPsiFlat[i]->SetYTitle(Form("Counts (cent<80%c)",'%'));
    for(int j = 0; j<NumCentBins; j++) {
      TString hname = Form("psiFlat_%d_%d",(int) wcent[j],(int) wcent[j+1]);
      hPsiFlatCent[i][j] = subdir.make<TH1D>(hname.Data(),hname.Data(),800,-psirange,psirange);
      hPsiFlatCent[i][j]->SetXTitle("#Psi");
      hPsiFlatCent[i][j]->SetYTitle(Form("Counts (%d<cent#leq%d%c)",(int) wcent[j],(int) wcent[j+1],'%'));
    }  

  }
  
}


HiEvtPlaneFlatCalib::~HiEvtPlaneFlatCalib()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
HiEvtPlaneFlatCalib::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;
  //
  //Get Flattening Parameters
  //
  if(genFlatPsi_ && FirstEvent) {
    FirstEvent = kFALSE;
    edm::ESHandle<RPFlatParams> flatparmsDB_;
    iSetup.get<HeavyIonRPRcd>().get(flatparmsDB_);
    int flatTableSize = flatparmsDB_->m_table.size();
    for(int i = 0; i<flatTableSize; i++) {
      const RPFlatParams::EP* thisBin = &(flatparmsDB_->m_table[i]);
      for(int j = 0; j<NumEPNames; j++) {
	int indx = thisBin->RPNameIndx[j];
	if(indx>=0) {
	  flat[indx]->SetXDB(i, thisBin->x[j]);
	  flat[indx]->SetYDB(i, thisBin->y[j]);
	  flatXDBhist[indx]->SetBinContent(i+1,thisBin->x[j]);
	  flatYDBhist[indx]->SetBinContent(i+1,thisBin->y[j]);
	}
      }
    }
  }
  //
  //Get Centrality
  //

  edm::Handle<int> ch;
  iEvent.getByLabel(centrality_,ch);
  int bin = *(ch.product());
  double centval = 2.5*bin+1.25;
  hcent->Fill(bin);
  //
  //Get Vertex
  //
  edm::Handle<reco::VertexCollection> vertexCollection3;
  iEvent.getByLabel("hiSelectedVertex",vertexCollection3);
  const reco::VertexCollection * vertices3 = vertexCollection3.product();
  vs_sell = vertices3->size();
  if(vs_sell>0) {
    vzr_sell = vertices3->begin()->z();
    vzErr_sell = vertices3->begin()->zError();
  } else
    vzr_sell = -999.9;
    
  //
  //Get Event Planes
  //

  Handle<reco::EvtPlaneCollection> evtPlanes;
  iEvent.getByLabel(inputPlanes_,evtPlanes);

  if(!evtPlanes.isValid()){
    //cout << "Error! Can't get hiEvtPlane product!" << endl;
    return ;
  }
  double psiFull[NumEPNames];
  for(int i = 0; i<NumEPNames; i++) {
    psiFull[i] = -10;
  }

  EvtPlane * ep[NumEPNames];
  for(int i = 0; i<NumEPNames; i++) {
    ep[i]=0;
  }
  for (EvtPlaneCollection::const_iterator rp = evtPlanes->begin();rp !=evtPlanes->end(); rp++) {
    string tmpName = rp->label();
    if(rp->angle() > -5) {
      string baseName = rp->label();
      for(int i = 0; i< NumEPNames; i++) {
	if(EPNames[i].compare(baseName)==0) {
	  double angorig = rp->angle();
	  double psiFlat = flat[i]->GetFlatPsi(angorig,vzr_sell,bin);
	  epang[i]=psiFlat;
	  if(EPNames[i].compare(rp->label())==0) {
	    flat[i]->Fill(angorig,vzr_sell,bin);
	    if(i==0)  hvtx->Fill(vzr_sell);

	    if(centval<=80) hPsi[i]->Fill(angorig);
	    psiFull[i] = psiFlat;
	    if(genFlatPsi_) {
	      if(centval<=80) hPsiFlat[i]->Fill(psiFlat);
	      for(int j = 0; j<NumCentBins; j++) {
		if(centval>wcent[j]&&centval<=wcent[j+1]) hPsiFlatCent[i][j]->Fill(psiFlat);
	      }
	    }
	  } 
	}
      }
    }    
  }

}

// ------------ method called once each job just before starting event loop  ------------
void 
HiEvtPlaneFlatCalib::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HiEvtPlaneFlatCalib::endJob() {
  for(int i = 0; i<NumEPNames; i++) {
    for(int j = 0; j<flat[i]->GetHBins();j++) {
      flatXhist[i]->SetBinContent(j+1,flat[i]->GetX(j));
      flatYhist[i]->SetBinContent(j+1,flat[i]->GetY(j));
      flatCnthist[i]->SetBinContent(j+1,flat[i]->GetCnt(j));
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiEvtPlaneFlatCalib);
