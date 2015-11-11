// ----------------------------------------------------------------------
// MCVerticesAnalyzer
// ---------

#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <bitset>

#include "HeavyIonsAnalysis/VertexAnalysis/interface/MCVerticesAnalyzer.h"

#include "CondFormats/Alignment/interface/Definitions.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

DEFINE_FWK_MODULE(MCVerticesAnalyzer);

#include <TROOT.h>
#include <TSystem.h>
#include <TTree.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>

using namespace std;
using namespace edm;
using namespace reco;

// ----------------------------------------------------------------------
MCVerticesAnalyzer::MCVerticesAnalyzer(edm::ParameterSet const& iConfig): 
    fPrimaryVertexCollectionLabel(iConfig.getUntrackedParameter<InputTag>("vertexCollLabel", edm::InputTag("offlinePrimaryVertices"))), 
    fPileUpInfoLabel(edm::InputTag("addPileupInfo"))
{

    edm::Service<TFileService> fs;

    tree = fs->make<TTree>("tree","mcvertices");
    
    pileup = fs->make<TH1F>("pileup","pileup",100,0,100);
 
    tree->Branch("nGoodVtx","map<int,int>",&nGoodVtx); 
    tree->Branch("nValidVtx","map<int,int>",&nValidVtx); 
    recoVtxToken=consumes<reco::VertexCollection>(fPrimaryVertexCollectionLabel);
    tree->Branch("nVtx",&nVtx,"nVtx/I");
    tree->Branch("vtx_nTrk",&vtx_nTrk,"vtx_nTrk[nVtx]/I");
    tree->Branch("vtx_ndof",&vtx_ndof,"vtx_ndof[nVtx]/I");
    tree->Branch("vtx_x",&vtx_x,"vtx_x[nVtx]/F");
    tree->Branch("vtx_y",&vtx_y,"vtx_y[nVtx]/F");
    tree->Branch("vtx_z",&vtx_z,"vtx_z[nVtx]/F");
    tree->Branch("vtx_xError",&vtx_xError,"vtx_xError[nVtx]/F");
    tree->Branch("vtx_yError",&vtx_yError,"vtx_yError[nVtx]/F");
    tree->Branch("vtx_zError",&vtx_zError,"vtx_zError[nVtx]/F");
    tree->Branch("vtx_chi2",&vtx_chi2,"vtx_chi2[nVtx]/F");
    tree->Branch("vtx_normchi2",&vtx_normchi2,"vtx_normchi2[nVtx]/F");
    tree->Branch("vtx_isValid",&vtx_isValid,"vtx_isValid[nVtx]/O");
    tree->Branch("vtx_isFake",&vtx_isFake,"vtx_isFake[nVtx]/O");
    tree->Branch("vtx_isGood",&vtx_isGood,"vtx_isGood[nVtx]/O");
    
    pileUpToken=consumes<std::vector< PileupSummaryInfo> >(fPileUpInfoLabel);
    tree->Branch("nPU",&nPU,"nPU/I");
}

// ----------------------------------------------------------------------
MCVerticesAnalyzer::~MCVerticesAnalyzer() { 
}  

// ----------------------------------------------------------------------
void MCVerticesAnalyzer::endJob() { 
 
}

// ----------------------------------------------------------------------
void MCVerticesAnalyzer::beginJob() {

}

// ----------------------------------------------------------------------
void MCVerticesAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& isetup){
  Reset();
}

// ----------------------------------------------------------------------
void MCVerticesAnalyzer::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& isetup){
  // tree->Fill();
}


// ----------------------------------------------------------------------
void MCVerticesAnalyzer::analyze(const edm::Event& iEvent, 
            const edm::EventSetup& iSetup)  {

  using namespace edm;
  using reco::VertexCollection;

  Reset();
  
  eventCounter++;

  int bunchCrossing   = iEvent.bunchCrossing();

  edm::Handle<std::vector< PileupSummaryInfo> > pileUpInfo;
  if(iEvent.getByToken(pileUpToken, pileUpInfo)) {
    std::vector<PileupSummaryInfo>::const_iterator PVI;
    for(PVI = pileUpInfo->begin(); PVI != pileUpInfo->end(); ++PVI) {
      int pu_bunchcrossing = PVI->getBunchCrossing();
      if(pu_bunchcrossing == 0) {
        nPU=PVI->getPU_NumInteractions();
        pileup->Fill(nPU);
      }
    }
  }
  
  edm::Handle<reco::VertexCollection> recVtxs;
  if(iEvent.getByToken(recoVtxToken,recVtxs)) {
        
    if(recVtxs.isValid()){
      int ivtx=0;
      for(reco::VertexCollection::const_iterator v=recVtxs->begin(); v!=recVtxs->end(); ++v){
        if(v->isFake()) continue;
        vtx_isGood[ivtx] = false;
        vtx_nTrk[ivtx] = v->tracksSize();
        vtx_ndof[ivtx] = (int)v->ndof();
        vtx_x[ivtx] = v->x();
        vtx_y[ivtx] = v->y();
        vtx_z[ivtx] = v->z();
        vtx_xError[ivtx] = v->xError();
        vtx_yError[ivtx] = v->yError();
        vtx_zError[ivtx] = v->zError();
        vtx_chi2[ivtx] = v->chi2();
        vtx_normchi2[ivtx] = v->normalizedChi2();
        vtx_isValid[ivtx] = v->isValid();
        vtx_isFake[ivtx] = v->isFake();
        if(vtx_isValid[ivtx] && (vtx_isFake[ivtx] == 0)){
          nValidVtx[bunchCrossing]=nValidVtx[bunchCrossing]+1;
        }
        if(vtx_ndof[ivtx] > 4 && vtx_isValid[ivtx] && (vtx_isFake[ivtx] == 0)){
          if(vtx_nTrk[ivtx] > 0){
            nGoodVtx[bunchCrossing]=nGoodVtx[bunchCrossing]+1;
            vtx_isGood[ivtx] = true;
          }
        }
        ivtx++;
      }
      nVtx=ivtx;
    }
  }
  tree->Fill();
  Reset();
}


void MCVerticesAnalyzer::Reset() {
    nVtx = 0;
    nValidVtx.clear();
    nGoodVtx.clear();
    eventCounter=1;
}
