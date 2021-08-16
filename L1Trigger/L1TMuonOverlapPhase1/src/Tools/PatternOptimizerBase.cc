/*
 * PatternOptimizerBase.cc
 *
 *  Created on: Oct 17, 2018
 *      Author: kbunkow
 */
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/PatternOptimizerBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/AlgoMuon.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPattern.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternResult.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/XMLConfigWriter.h"
#include "L1Trigger/RPCTrigger/interface/RPCConst.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/Track/interface/CoreSimTrack.h"
#include "SimDataFormats/Track/interface/SimTrack.h"

#include "Math/VectorUtil.h"

#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TStyle.h"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

double PatternOptimizerBase::vxMuRate(double pt_GeV) {
  if (pt_GeV == 0)
    return 0.0;
  const double lum = 2.0e34;  //defoult is 1.0e34;
  const double dabseta = 1.0;
  const double dpt = 1.0;
  const double afactor = 1.0e-34 * lum * dabseta * dpt;
  const double a = 2 * 1.3084E6;
  const double mu = -0.725;
  const double sigma = 0.4333;
  const double s2 = 2 * sigma * sigma;

  double ptlog10;
  ptlog10 = log10(pt_GeV);
  double ex = (ptlog10 - mu) * (ptlog10 - mu) / s2;
  double rate = (a * exp(-ex) * afactor);
  //edm::LogError("RPCTrigger")<<ptCode<<" "<<rate;//<<<<<<<<<<<<<<<<<<<<<<<<
  return rate;
}

double PatternOptimizerBase::vxIntegMuRate(double pt_GeV, double dpt, double etaFrom, double etaTo) {
  //calkowanie metoda trapezow - nie do konca dobre
  double rate = 0.5 * (vxMuRate(pt_GeV) + vxMuRate(pt_GeV + dpt)) * dpt;

  rate = rate * (etaTo - etaFrom);
  //edm::LogError("RPCTrigger")<<ptCode<<" "<<rate;//<<<<<<<<<<<<<<<<<<<<<<<<
  return rate;
}

PatternOptimizerBase::PatternOptimizerBase(const edm::ParameterSet& edmCfg, const OMTFConfiguration* omtfConfig)
    : edmCfg(edmCfg), omtfConfig(omtfConfig), simMuon(nullptr) {
  // TODO Auto-generated constructor stub

  simMuPt = new TH1I("simMuPt", "simMuPt", goldenPatterns.size(), -0.5, goldenPatterns.size() - 0.5);
  simMuFoundByOmtfPt =
      new TH1I("simMuFoundByOmtfPt", "simMuFoundByOmtfPt", goldenPatterns.size(), -0.5, goldenPatterns.size() - 0.5);

  simMuPtSpectrum = new TH1F("simMuPtSpectrum", "simMuPtSpectrum", 800, 0, 400);
}

PatternOptimizerBase::PatternOptimizerBase(const edm::ParameterSet& edmCfg,
                                           const OMTFConfiguration* omtfConfig,
                                           std::vector<std::shared_ptr<GoldenPatternWithStat> >& gps)
    : edmCfg(edmCfg), omtfConfig(omtfConfig), goldenPatterns(gps), simMuon(nullptr) {
  // TODO Auto-generated constructor stub

  simMuPt = new TH1I("simMuPt", "simMuPt", goldenPatterns.size(), -0.5, goldenPatterns.size() - 0.5);
  simMuFoundByOmtfPt =
      new TH1I("simMuFoundByOmtfPt", "simMuFoundByOmtfPt", goldenPatterns.size(), -0.5, goldenPatterns.size() - 0.5);

  simMuPtSpectrum = new TH1F("simMuPtSpectrum", "simMuPtSpectrum", 800, 0, 400);

  if (edmCfg.exists("simTracksTag") == false)
    edm::LogError("l1tOmtfEventPrint")  << "simTracksTag not found !!!"<<std::endl;
}

PatternOptimizerBase::~PatternOptimizerBase() {
  // TODO Auto-generated destructor stub
}

void PatternOptimizerBase::printPatterns() {
  edm::LogVerbatim("l1tOmtfEventPrint")  << __FUNCTION__ << ": " << __LINE__ << " called!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << std::endl;
  for (int patNum = goldenPatterns.size() - 1; patNum >= 0; patNum--) {
    double pt = omtfConfig->getPatternPtRange(patNum).ptFrom;
    if (pt > 0) {
      edm::LogVerbatim("l1tOmtfEventPrint")  << "cmsRun runThresholdCalc.py " << patNum << " " << (patNum + 1) << " _" << RPCConst::iptFromPt(pt) << "_";
      if (goldenPatterns[patNum]->key().theCharge == -1)
        edm::LogVerbatim("l1tOmtfEventPrint")  << "m_";
      else
        edm::LogVerbatim("l1tOmtfEventPrint")  << "p_";

      edm::LogVerbatim("l1tOmtfEventPrint")  << " > out" << patNum << ".txt" << std::endl;
    }
  }
}

void PatternOptimizerBase::observeProcesorEmulation(unsigned int iProcessor,
                                                    l1t::tftype mtfType,
                                                    const std::shared_ptr<OMTFinput>& input,
                                                    const AlgoMuons& algoCandidates,
                                                    const AlgoMuons& gbCandidates,
                                                    const std::vector<l1t::RegionalMuonCand>& candMuons) {
  unsigned int procIndx = omtfConfig->getProcIndx(iProcessor, mtfType);

  /*
  double ptSim = simMuon->momentum().pt();
  int chargeSim = (abs(simMuon->type()) == 13) ? simMuon->type()/-13 : 0;
  int patNum = omtfConfig->getPatternNum(ptSim, chargeSim);
  GoldenPatternWithStat* exptCandGp = goldenPatterns.at(patNum).get(); // expected pattern
*/

  //bool found = false;

  unsigned int i = 0;
  for (auto& gbCandidate : gbCandidates) {
    //int iRefHit = gbCandidate.getRefHitNumber();
    if (gbCandidate->getGoldenPatern() != nullptr &&
        gbCandidate->getGpResult().getFiredLayerCnt() > omtfCand->getGpResult().getFiredLayerCnt()) {
      //edm::LogVerbatim("l1tOmtfEventPrint") <<__FUNCTION__<<":"<<__LINE__<<" gbCandidate "<<gbCandidate<<" "<<std::endl;
      omtfCand = gbCandidate;
      //omtfResult = gbCandidate.getGoldenPatern()->getResults()[procIndx][iRefHit]; //TODO be carrefful, because in principle the results sored by the goldenPattern can be altered in one event. In phae I omtf this should not happened, but in OMTFProcessorTTMerger - yes
      //exptResult = exptCandGp->getResults()[procIndx][iRefHit];
      candProcIndx = procIndx;

      regionalMuonCand = candMuons.at(
          i);  //should be good, as the regionalMuonCand is created for every  gbCandidate in OMTFProcessor<GoldenPatternType>::getFinalcandidates
      //found = true;

      this->algoCandidates = algoCandidates;
    }
    i++;
  }

  //////////////////////debug printout/////////////////////////////
  /*if(found) {
    GoldenPatternWithStat* omtfCandGp = static_cast<GoldenPatternWithStat*>(omtfCand.getGoldenPatern());
    if( omtfCandGp->key().thePt > 100 && exptCandGp->key().thePt <= 15 ) {
      //edm::LogVerbatim("l1tOmtfEventPrint") <<iEvent.id()<<std::endl;
      cout<<" ptSim "<<ptSim<<" chargeSim "<<chargeSim<<std::endl;
      edm::LogVerbatim("l1tOmtfEventPrint") <<"iProcessor "<<iProcessor<<" exptCandGp "<<exptCandGp->key()<<std::endl;
      edm::LogVerbatim("l1tOmtfEventPrint") <<"iProcessor "<<iProcessor<<" omtfCandGp "<<omtfCandGp->key()<<std::endl;
      edm::LogVerbatim("l1tOmtfEventPrint") <<"omtfResult "<<std::endl<<omtfResult<<std::endl;
      int refHitNum = omtfCand.getRefHitNumber();
      edm::LogVerbatim("l1tOmtfEventPrint") <<"other gps results"<<endl;
      for(auto& gp : goldenPatterns) {
        if(omtfResult.getFiredLayerCnt() == gp->getResults()[procIndx][iRefHit].getFiredLayerCnt() )
        {
          edm::LogVerbatim("l1tOmtfEventPrint") <<gp->key()<<std::endl<<gp->getResults()[procIndx][iRefHit]<<std::endl;
        }
      }
      std::cout<<std::endl;
    }
  }*/
}

void PatternOptimizerBase::observeEventBegin(const edm::Event& iEvent) {
  omtfCand.reset(new AlgoMuon());
  candProcIndx = 0xffff;
  //exptResult =  GoldenPatternResult();

  simMuon = findSimMuon(iEvent);
  //edm::LogVerbatim("l1tOmtfEventPrint") <<__FUNCTION__<<":"<<__LINE__<<" evevt "<<iEvent.id().event()<<" simMuon pt "<<simMuon->momentum().pt()<<" GeV "<<std::endl;
}

void PatternOptimizerBase::observeEventEnd(const edm::Event& iEvent,
                                           std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) {
  if (simMuon == nullptr || omtfCand->getGoldenPatern() == nullptr)  //no sim muon or empty candidate
    return;

  double ptSim = simMuon->momentum().pt();
  int chargeSim = (abs(simMuon->type()) == 13) ? simMuon->type() / -13 : 0;

  unsigned int exptPatNum = omtfConfig->getPatternNum(ptSim, chargeSim);
  GoldenPatternWithStat* exptCandGp = goldenPatterns.at(exptPatNum).get();  // expected pattern
  simMuFoundByOmtfPt->Fill(exptCandGp->key().theNumber);                    //TODO add weight of the muons pt spectrum

  simMuPtSpectrum->Fill(ptSim, getEventRateWeight(ptSim));
}

void PatternOptimizerBase::endJob() {
  std::string fName = edmCfg.getParameter<std::string>("optimisedPatsXmlFile");
  edm::LogImportant("PatternOptimizer") << " Writing optimized patterns to " << fName << std::endl;
  XMLConfigWriter xmlWriter(omtfConfig, false, false);
  xmlWriter.writeGPs(goldenPatterns, fName);

  fName.replace(fName.find('.'), fName.length(), ".root");
  savePatternsInRoot(fName);
}

const SimTrack* PatternOptimizerBase::findSimMuon(const edm::Event& event, const SimTrack* previous) {
  const SimTrack* result = nullptr;
  if (edmCfg.exists("simTracksTag") == false)
    return result;

  edm::Handle<edm::SimTrackContainer> simTks;
  event.getByLabel(edmCfg.getParameter<edm::InputTag>("simTracksTag"), simTks);

  //LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<" simTks->size() "<<simTks->size()<<std::endl;
  for (std::vector<SimTrack>::const_iterator it = simTks->begin(); it < simTks->end(); it++) {
    const SimTrack& aTrack = *it;
    if (!(aTrack.type() == 13 || aTrack.type() == -13))
      continue;
    if (previous && ROOT::Math::VectorUtil::DeltaR(aTrack.momentum(), previous->momentum()) < 0.07)
      continue;
    if (!result || aTrack.momentum().pt() > result->momentum().pt())
      result = &aTrack;
  }
  return result;
}

void PatternOptimizerBase::savePatternsInRoot(std::string rootFileName) {
  gStyle->SetOptStat(111111);
  TFile outfile(rootFileName.c_str(), "RECREATE");
  edm::LogVerbatim("l1tOmtfEventPrint") << __FUNCTION__ << ": " << __LINE__ << " out fileName " << rootFileName << " outfile->GetName() "
       << outfile.GetName() << " writeLayerStat " << writeLayerStat << endl;

  outfile.cd();
  simMuFoundByOmtfPt->Write();

  simMuPtSpectrum->Write();

  outfile.mkdir("patternsPdfs")->cd();
  outfile.mkdir("patternsPdfs/canvases");
  outfile.mkdir("layerStats");
  ostringstream ostrName;
  ostringstream ostrTtle;
  vector<TH1F*> classProbHists;
  for (unsigned int iRefLayer = 0; iRefLayer < goldenPatterns[0]->getPdf()[0].size(); ++iRefLayer) {
    ostrName.str("");
    ostrName << "Neg_RefLayer_" << iRefLayer;
    ostrTtle.str("");
    ostrTtle << "Neg_RefLayer_" << iRefLayer;
    classProbHists.push_back(new TH1F(
        ostrName.str().c_str(), ostrTtle.str().c_str(), goldenPatterns.size(), -0.5, goldenPatterns.size() - 0.5));

    ostrName.str("");
    ostrName << "Pos_RefLayer_" << iRefLayer;
    ostrTtle.str("");
    ostrTtle << "Pos_RefLayer_" << iRefLayer;
    classProbHists.push_back(new TH1F(
        ostrName.str().c_str(), ostrTtle.str().c_str(), goldenPatterns.size(), -0.5, goldenPatterns.size() - 0.5));
  }

  for (auto& gp : goldenPatterns) {
    OMTFConfiguration::PatternPt patternPt = omtfConfig->getPatternPtRange(gp->key().theNumber);
    if (gp->key().thePt == 0)
      continue;
    //edm::LogVerbatim("l1tOmtfEventPrint") <<__FUNCTION__<<": "<<__LINE__<<" "<<gp->key()<<std::endl;
    ostrName.str("");
    ostrName << "PatNum_" << gp->key().theNumber;
    ostrTtle.str("");
    ostrTtle << "PatNum_" << gp->key().theNumber << "_ptCode_" << gp->key().thePt << "_Pt_" << patternPt.ptFrom << "_"
             << patternPt.ptTo << "_GeV";
    TCanvas* canvas = new TCanvas(ostrName.str().c_str(), ostrTtle.str().c_str(), 1200, 1000);
    canvas->Divide(gp->getPdf().size(), gp->getPdf()[0].size(), 0, 0);

    for (unsigned int iLayer = 0; iLayer < gp->getPdf().size(); ++iLayer) {
      for (unsigned int iRefLayer = 0; iRefLayer < gp->getPdf()[iLayer].size(); ++iRefLayer) {
        canvas->cd(1 + iLayer + iRefLayer * gp->getPdf().size());
        //unsigned int refLayerLogicNumber = omtfConfig->getRefToLogicNumber()[iRefLayer];
        ostrName.str("");
        ostrName << "PatNum_" << gp->key().theNumber << "_refLayer_" << iRefLayer << "_Layer_" << iLayer;
        ostrTtle.str("");
        ostrTtle << "PatNum " << gp->key().theNumber << " ptCode " << gp->key().thePt << " refLayer " << iRefLayer
                 << " Layer " << iLayer << " meanDistPhi " << gp->meanDistPhi[iLayer][iRefLayer][0]
                 << " distPhiBitShift "
                 << gp->getDistPhiBitShift(iLayer, iRefLayer);  //"_Pt_"<<patternPt.ptFrom<<"_"<<patternPt.ptTo<<"_GeV
        //edm::LogVerbatim("l1tOmtfEventPrint") <<__FUNCTION__<<": "<<__LINE__<<" creating hist "<<ostrTtle.str()<<std::endl;
        TH1F* hist = new TH1F(
            ostrName.str().c_str(), ostrTtle.str().c_str(), omtfConfig->nPdfBins(), -0.5, omtfConfig->nPdfBins() - 0.5);
        for (unsigned int iPdf = 0; iPdf < gp->getPdf()[iLayer][iRefLayer].size(); iPdf++) {
          hist->Fill(iPdf, gp->pdfAllRef[iLayer][iRefLayer][iPdf]);
        }
        if ((int)iLayer == (omtfConfig->getRefToLogicNumber()[iRefLayer]))
          hist->SetLineColor(kGreen);

        hist->GetYaxis()->SetRangeUser(0, omtfConfig->pdfMaxValue() + 1);
        hist->Draw("hist");

        outfile.cd("patternsPdfs");
        hist->Write();

        /////////////////////// histLayerStat
        if (writeLayerStat) {
          string histName = "histLayerStat_" + ostrName.str();
          unsigned int binCnt = gp->getStatistics()[iLayer][iRefLayer].size();
          TH1I* histLayerStat = new TH1I(histName.c_str(), histName.c_str(), binCnt, -0.5, binCnt - 0.5);
          for (unsigned int iBin = 0; iBin < binCnt; iBin++) {
            histLayerStat->Fill(iBin, gp->getStatistics()[iLayer][iRefLayer][iBin][0]);
          }

          outfile.cd("layerStats");
          histLayerStat->Write();
        }
      }
    }
    outfile.cd("patternsPdfs/canvases");
    canvas->Write();
    delete canvas;

    unsigned int iPdf = omtfConfig->nPdfBins() / 2;
    for (unsigned int iRefLayer = 0; iRefLayer < gp->getPdf()[0].size(); ++iRefLayer) {
      unsigned int refLayerLogicNumber = omtfConfig->getRefToLogicNumber()[iRefLayer];
      if (gp->key().theCharge == -1) {
        classProbHists[2 * iRefLayer]->Fill(gp->key().theNumber, gp->pdfAllRef[refLayerLogicNumber][iRefLayer][iPdf]);
      } else
        classProbHists[2 * iRefLayer + 1]->Fill(gp->key().theNumber,
                                                gp->pdfAllRef[refLayerLogicNumber][iRefLayer][iPdf]);
    }
  }

  outfile.mkdir("patternsPdfSumStat")->cd();
  for (auto& gp : goldenPatterns) {
    for (unsigned int iRefLayer = 0; iRefLayer < gp->gpProbabilityStat.size(); ++iRefLayer) {
      gp->gpProbabilityStat[iRefLayer]->Write();
    }
  }

  outfile.cd();
  for (auto& classProbHist : classProbHists) {
    classProbHist->Write();
  }

  saveHists(outfile);

  outfile.Close();
}
