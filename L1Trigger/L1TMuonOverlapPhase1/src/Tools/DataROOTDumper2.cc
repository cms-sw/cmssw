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
#include "TParameter.h"
#include "TObjString.h"

#include <boost/range/adaptor/reversed.hpp>
#include <boost/timer/timer.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

DataROOTDumper2::DataROOTDumper2(const edm::ParameterSet& edmCfg,
                                 const OMTFConfiguration* omtfConfig,
                                 const std::vector<std::shared_ptr<GoldenPattern> >& gps,
                                 std::string rootFileName)
    : PatternOptimizerBase(edmCfg, omtfConfig), gps(gps), event(omtfConfig->nTestRefHits(), gps.size()) {
  edm::LogVerbatim("l1tOmtfEventPrint") << " gps.size() " << gps.size() << " omtfConfig->nTestRefHits() "
                                        << omtfConfig->nTestRefHits() << " event.omtfGpResultsPdfSum.num_elements() "
                                        << event.omtfGpResultsPdfSum.num_elements() << endl;
  initializeTTree(rootFileName);

  if (false) {                                           //TODO!!!!!!!!!!!!
    gpResultsToPt = new GpResultsToPt(gps, omtfConfig);  //TODO move to processor

    std::string fileName = edmCfg.getParameter<std::string>("gpResultsToPtFile");
    std::ifstream ifs(fileName);

    boost::archive::text_iarchive inArch(ifs);
    //boost::archive::text_oarchive txtOutArch(ofs);

    //const PdfModule* pdfModuleImpl = dynamic_cast<const PdfModule*>(pdfModule);
    // write class instance to archive
    edm::LogImportant("l1tOmtfEventPrint")
        << __FUNCTION__ << ": " << __LINE__ << " writing gpResultsToPt to file " << fileName << std::endl;
    inArch >> *gpResultsToPt;
  }

  edm::LogVerbatim("l1tOmtfEventPrint")  << " DataROOTDumper2 created"<<std::endl;
}

DataROOTDumper2::~DataROOTDumper2() { saveTTree(); }

void DataROOTDumper2::initializeTTree(std::string rootFileName) {
  rootFile = new TFile(rootFileName.c_str(), "RECREATE");
  rootTree = new TTree("OMTFHitsTree", "");

  rootTree->Branch("muonPt", &event.muonPt);
  rootTree->Branch("muonEta", &event.muonEta);
  rootTree->Branch("muonPhi", &event.muonPhi);
  rootTree->Branch("muonCharge", &event.muonCharge);

  rootTree->Branch("omtfPt", &event.omtfPt);
  rootTree->Branch("omtfEta", &event.omtfEta);
  rootTree->Branch("omtfPhi", &event.omtfPhi);
  rootTree->Branch("omtfCharge", &event.omtfCharge);

  rootTree->Branch("omtfScore", &event.omtfScore);
  rootTree->Branch("omtfQuality", &event.omtfQuality);
  rootTree->Branch("omtfRefLayer", &event.omtfRefLayer);
  rootTree->Branch("omtfProcessor", &event.omtfProcessor);

  rootTree->Branch("omtfFiredLayers", &event.omtfFiredLayers);  //<<<<<<<<<<<<<<<<<<<<<<!!!!TODOO

  if (gpResultsToPt)
    rootTree->Branch("omtfPtCont", &event.omtfPtCont);

  rootTree->Branch("hits", &event.hits);

  if (dumpGpResults) {  //TODO not finished, probably has no sense

    unsigned int elementCnt = event.omtfGpResultsPdfSum.num_elements();
    rootTree->Branch("omtfGpResultsPdfSum",
                     event.omtfGpResultsPdfSum.data(),
                     ("omtfGpResultsPdfSum[" + to_string(elementCnt) + "]/F").c_str());

    elementCnt = event.omtfGpResultsPdfSum.num_elements();
    rootTree->Branch("omtfGpResultsFiredLayers",
                     event.omtfGpResultsFiredLayers.data(),
                     ("omtfGpResultsFiredLayers[" + to_string(elementCnt) + "]/F").c_str());

    edm::LogVerbatim("l1tOmtfEventPrint") << " dumpGpResults "
                                          << " omtfGpResultsFiredLayers elementCnt " << elementCnt << std::endl;

    //rootTree->GetUserInfo()->Add(new TParameter<unsigned int>("elementCnt", elementCnt));
    rootTree->GetUserInfo()->Add(new TObjString(("elementCnt:" + to_string(elementCnt)).c_str()));
  }

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

  //PatternOptimizerBase::observeEventEnd(iEvent, finalCandidates); not needed

  event.muonPt = simMuon->momentum().pt();
  event.muonEta = simMuon->momentum().eta();

  /*  if(abs(event.muonEta) < 0.8 || abs(event.muonEta) > 1.24)
    return;*/

  event.muonPhi = simMuon->momentum().phi();
  event.muonCharge = muonCharge;  //TODO

  if (omtfCand->getPt() > 0) {                     //&& omtfCand->getFiredLayerCnt() > 3
    event.omtfPt = (omtfCand->getPt() - 1) / 2.0;  //TODO check
    event.omtfEta = omtfCand->getEtaHw() / 240. * 2.61;
    event.omtfPhi = omtfCand->getPhi();
    event.omtfCharge = std::pow(-1, regionalMuonCand.hwSign());
    event.omtfScore = omtfCand->getPdfSum();
    event.omtfQuality = regionalMuonCand.hwQual();  //omtfCand->getQ();
    event.omtfFiredLayers = omtfCand->getFiredLayerBits();
    event.omtfRefLayer = omtfCand->getRefLayer();
    event.omtfProcessor = candProcIndx;

    if (gpResultsToPt)
      event.omtfPtCont = omtfConfig->hwPtToGev(gpResultsToPt->getValue(omtfCand, candProcIndx));

    event.hits.clear();

    //unsigned int iRefHit = omtfCand->getRefHitNumber();

    auto& gpResult = omtfCand->getGpResult();
    //int pdfMiddle = 1<<(omtfConfig->nPdfAddrBits()-1);

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

        /*int phiDist = stubResult.getPdfBin() - pdfMiddle;

        phiDist = (phiDist << omtfCand->getGoldenPatern()->getDistPhiBitShift(iLogicLayer, omtfCand->getRefLayer()) );
        phiDist += (omtfCand->getGoldenPatern()->meanDistPhiValue(iLogicLayer, omtfCand->getRefLayer()) ); //removing the shift applied in the GoldenPatternBase::process1Layer1RefLayer
        //TODO include the phiDist = phiDist >> this->getDistPhiBitShift(iLayer, iRefLayer); applied in the GoldenPatternBase::process1Layer1RefLaye
        hit.phiDist = phiDist;*/

        int hitPhi = stubResult.getMuonStub()->phiHw;
        unsigned int refLayerLogicNum = omtfConfig->getRefToLogicNumber()[omtfCand->getRefLayer()];
        int phiRefHit = gpResult.getStubResults()[refLayerLogicNum].getMuonStub()->phiHw;

        if (omtfConfig->isBendingLayer(iLogicLayer)) {
          hitPhi = stubResult.getMuonStub()->phiBHw;
          phiRefHit = 0;  //phi ref hit for the banding layer set to 0, since it should not be included in the phiDist
        }

        //phiDist = hitPhi - phiRefHit;
        hit.phiDist = hitPhi - phiRefHit;

        /* edm::LogVerbatim("l1tOmtfEventPrint")<<" muonPt "<<event.muonPt<<" omtfPt "<<event.omtfPt<<" RefLayer "<<event.omtfRefLayer
            <<" layer "<<int(hit.layer)<<" PdfBin "<<stubResult.getPdfBin()<<" hit.phiDist "<<hit.phiDist<<" valid "<<stubResult.getValid()<<" " //<<" phiDist "<<phiDist
            <<" getDistPhiBitShift "<<omtfCand->getGoldenPatern()->getDistPhiBitShift(iLogicLayer, omtfCand->getRefLayer())
            <<" meanDistPhiValue   "<<omtfCand->getGoldenPatern()->meanDistPhiValue(iLogicLayer, omtfCand->getRefLayer())//<<(phiDist != hit.phiDist? "!!!!!!!<<<<<" : "")
            <<endl;*/

        if (hit.phiDist > 504 || hit.phiDist < -512) {
          edm::LogVerbatim("l1tOmtfEventPrint")
              << " muonPt " << event.muonPt << " omtfPt " << event.omtfPt << " RefLayer " << event.omtfRefLayer
              << " layer " << int(hit.layer) << " hit.phiDist " << hit.phiDist << " valid " << stubResult.getValid()
              << " !!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        }

        DetId detId(stubResult.getMuonStub()->detId);
        if (detId.subdetId() == MuonSubdetId::CSC) {
          CSCDetId cscId(detId);
          hit.z = cscId.chamber() % 2;
        }

        event.hits.push_back(hit.rawData);
      }
    }

    /*if( (int)event.hits.size() != omtfCand->getQ()) {
      edm::LogVerbatim("l1tOmtfEventPrint")<<" muonPt "<<event.muonPt<<" omtfPt "<<event.omtfPt<<" RefLayer "<<event.omtfRefLayer
                    <<" hits.size "<<event.hits.size()<<" omtfCand->getQ "<<omtfCand->getQ()<<" !!!!!!!!!!!!!!!!!!aaa!!!!!!"<<endl;
    }*/

    if (dumpGpResults) {  //TODO not finished, probably has no sense
      for (unsigned int iRefHit = 0; iRefHit < omtfConfig->nTestRefHits(); iRefHit++) {
        unsigned int iGP = 0;
        for (auto& itGP : gps) {
          if (itGP->key().thePt == 0)
            continue;

          auto& result = itGP->getResults()[candProcIndx][iRefHit];

          event.omtfGpResultsPdfSum[iRefHit][iGP] = result.getPdfSum();
          //result.getRefLayer();
          event.omtfGpResultsFiredLayers[iRefHit][iGP] = result.getFiredLayerBits();

          iGP++;
        }
      }
    }

    rootTree->Fill();
    evntCnt++;
  }
}

void DataROOTDumper2::endJob() { edm::LogVerbatim("l1tOmtfEventPrint") << " evntCnt " << evntCnt << endl; }
