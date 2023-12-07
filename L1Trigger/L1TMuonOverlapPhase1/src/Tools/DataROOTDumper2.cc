/*
 * DataROOTDumper2.cc
 *
 *  Created on: Dec 11, 2019
 *      Author: kbunkow
 */

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/DataROOTDumper2.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "TFile.h"
#include "TTree.h"

DataROOTDumper2::DataROOTDumper2(const edm::ParameterSet& edmCfg,
                                 const OMTFConfiguration* omtfConfig,
                                 std::string rootFileName)
    : EmulationObserverBase(edmCfg, omtfConfig) {
  edm::LogVerbatim("l1tOmtfEventPrint") << " omtfConfig->nTestRefHits() " << omtfConfig->nTestRefHits()
                                        << " event.omtfGpResultsPdfSum.num_elements() " << endl;
  initializeTTree(rootFileName);

  edm::LogVerbatim("l1tOmtfEventPrint") << " DataROOTDumper2 created" << std::endl;
}

DataROOTDumper2::~DataROOTDumper2() { saveTTree(); }

void DataROOTDumper2::initializeTTree(std::string rootFileName) {
  rootFile = new TFile(rootFileName.c_str(), "RECREATE");
  rootTree = new TTree("OMTFHitsTree", "");

  rootTree->Branch("muonPt", &omtfEvent.muonPt);
  rootTree->Branch("muonEta", &omtfEvent.muonEta);
  rootTree->Branch("muonPhi", &omtfEvent.muonPhi);
  rootTree->Branch("muonCharge", &omtfEvent.muonCharge);

  rootTree->Branch("omtfPt", &omtfEvent.omtfPt);
  rootTree->Branch("omtfEta", &omtfEvent.omtfEta);
  rootTree->Branch("omtfPhi", &omtfEvent.omtfPhi);
  rootTree->Branch("omtfCharge", &omtfEvent.omtfCharge);

  rootTree->Branch("omtfScore", &omtfEvent.omtfScore);
  rootTree->Branch("omtfQuality", &omtfEvent.omtfQuality);
  rootTree->Branch("omtfRefLayer", &omtfEvent.omtfRefLayer);
  rootTree->Branch("omtfProcessor", &omtfEvent.omtfProcessor);

  rootTree->Branch("omtfFiredLayers", &omtfEvent.omtfFiredLayers);  //<<<<<<<<<<<<<<<<<<<<<<!!!!TODOO

  ptGenPos = new TH1I("ptGenPos", "ptGenPos", 400, 0, 200);
  ptGenNeg = new TH1I("ptGenNeg", "ptGenNeg", 400, 0, 200);
}

void DataROOTDumper2::saveTTree() {
  rootFile->Write();
  ptGenPos->Write();
  ptGenNeg->Write();

  delete rootFile;
  delete rootTree;
}

void DataROOTDumper2::observeEventEnd(const edm::Event& iEvent,
                                      std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) {
  int muonCharge = 0;
  if (simMuon) {
    if (abs(simMuon->momentum().eta()) < 0.8 || abs(simMuon->momentum().eta()) > 1.24)
      return;

    muonCharge = (abs(simMuon->type()) == 13) ? simMuon->type() / -13 : 0;
    if (muonCharge > 0)
      ptGenPos->Fill(simMuon->momentum().pt());
    else
      ptGenNeg->Fill(simMuon->momentum().pt());
  }

  if (simMuon == nullptr || !omtfCand->isValid())  //no sim muon or empty candidate
    return;

  omtfEvent.muonPt = simMuon->momentum().pt();
  omtfEvent.muonEta = simMuon->momentum().eta();

  //TODO add cut on ete if needed
  /*  if(abs(event.muonEta) < 0.8 || abs(event.muonEta) > 1.24)
    return;*/

  omtfEvent.muonPhi = simMuon->momentum().phi();
  omtfEvent.muonCharge = muonCharge;  //TODO

  if (omtfCand->getPt() > 0) {  //&& omtfCand->getFiredLayerCnt() > 3 TODO add to the if needed
    omtfEvent.omtfPt = omtfConfig->hwPtToGev(omtfCand->getPt());
    omtfEvent.omtfEta = omtfConfig->hwEtaToEta(omtfCand->getEtaHw());
    omtfEvent.omtfPhi = omtfCand->getPhi();
    omtfEvent.omtfCharge = std::pow(-1, regionalMuonCand.hwSign());
    omtfEvent.omtfScore = omtfCand->getPdfSum();
    omtfEvent.omtfQuality = regionalMuonCand.hwQual();  //omtfCand->getQ();
    omtfEvent.omtfFiredLayers = omtfCand->getFiredLayerBits();
    omtfEvent.omtfRefLayer = omtfCand->getRefLayer();
    omtfEvent.omtfProcessor = candProcIndx;

    omtfEvent.hits.clear();

    auto& gpResult = omtfCand->getGpResult();

    /*
    edm::LogVerbatim("l1tOmtfEventPrint")<<"DataROOTDumper2:;observeEventEnd muonPt "<<event.muonPt<<" muonCharge "<<event.muonCharge
        <<" omtfPt "<<event.omtfPt<<" RefLayer "<<event.omtfRefLayer<<" omtfPtCont "<<event.omtfPtCont
        <<std::endl;
*/

    for (unsigned int iLogicLayer = 0; iLogicLayer < gpResult.getStubResults().size(); ++iLogicLayer) {
      auto& stubResult = gpResult.getStubResults()[iLogicLayer];
      if (stubResult.getMuonStub()) {  //&& stubResult.getValid() //TODO!!!!!!!!!!!!!!!!1
        OmtfEvent::Hit hit;
        hit.layer = iLogicLayer;
        hit.quality = stubResult.getMuonStub()->qualityHw;
        hit.eta = stubResult.getMuonStub()->etaHw;  //in which scale?
        hit.valid = stubResult.getValid();

        int hitPhi = stubResult.getMuonStub()->phiHw;
        unsigned int refLayerLogicNum = omtfConfig->getRefToLogicNumber()[omtfCand->getRefLayer()];
        int phiRefHit = gpResult.getStubResults()[refLayerLogicNum].getMuonStub()->phiHw;

        if (omtfConfig->isBendingLayer(iLogicLayer)) {
          hitPhi = stubResult.getMuonStub()->phiBHw;
          phiRefHit = 0;  //phi ref hit for the bending layer set to 0, since it should not be included in the phiDist
        }

        //phiDist = hitPhi - phiRefHit;
        hit.phiDist = hitPhi - phiRefHit;

        /* LogTrace("l1tOmtfEventPrint")<<" muonPt "<<event.muonPt<<" omtfPt "<<event.omtfPt<<" RefLayer "<<event.omtfRefLayer
            <<" layer "<<int(hit.layer)<<" PdfBin "<<stubResult.getPdfBin()<<" hit.phiDist "<<hit.phiDist<<" valid "<<stubResult.getValid()<<" " //<<" phiDist "<<phiDist
            <<" getDistPhiBitShift "<<omtfCand->getGoldenPatern()->getDistPhiBitShift(iLogicLayer, omtfCand->getRefLayer())
            <<" meanDistPhiValue   "<<omtfCand->getGoldenPatern()->meanDistPhiValue(iLogicLayer, omtfCand->getRefLayer())//<<(phiDist != hit.phiDist? "!!!!!!!<<<<<" : "")
            <<endl;*/

        if (hit.phiDist > 504 || hit.phiDist < -512) {
          edm::LogVerbatim("l1tOmtfEventPrint")
              << " muonPt " << omtfEvent.muonPt << " omtfPt " << omtfEvent.omtfPt << " RefLayer "
              << omtfEvent.omtfRefLayer << " layer " << int(hit.layer) << " hit.phiDist " << hit.phiDist << " valid "
              << stubResult.getValid() << " !!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        }

        DetId detId(stubResult.getMuonStub()->detId);
        if (detId.subdetId() == MuonSubdetId::CSC) {
          CSCDetId cscId(detId);
          hit.z = cscId.chamber() % 2;
        }

        omtfEvent.hits.push_back(hit.rawData);
      }
    }

    //debug
    /*if( (int)event.hits.size() != omtfCand->getQ()) {
      LogTrace("l1tOmtfEventPrint")<<" muonPt "<<event.muonPt<<" omtfPt "<<event.omtfPt<<" RefLayer "<<event.omtfRefLayer
                    <<" hits.size "<<event.hits.size()<<" omtfCand->getQ "<<omtfCand->getQ()<<" !!!!!!!!!!!!!!!!!!aaa!!!!!!"<<endl;
    }*/

    rootTree->Fill();
    evntCnt++;
  }
}

void DataROOTDumper2::endJob() { edm::LogVerbatim("l1tOmtfEventPrint") << " evntCnt " << evntCnt << endl; }
