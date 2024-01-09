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

#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TH2F.h"
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
  //edm::LogError("l1tOmtfEventPrint")<<ptCode<<" "<<rate;//<<<<<<<<<<<<<<<<<<<<<<<<
  return rate;
}

double PatternOptimizerBase::vxIntegMuRate(double pt_GeV, double dpt, double etaFrom, double etaTo) {
  //integration using trapeze method - not exact but good enough
  double rate = 0.5 * (vxMuRate(pt_GeV) + vxMuRate(pt_GeV + dpt)) * dpt;

  rate = rate * (etaTo - etaFrom);
  //edm::LogError("l1tOmtfEventPrint")<<ptCode<<" "<<rate;//<<<<<<<<<<<<<<<<<<<<<<<<
  return rate;
}

PatternOptimizerBase::PatternOptimizerBase(const edm::ParameterSet& edmCfg,
                                           const OMTFConfiguration* omtfConfig,
                                           GoldenPatternVec<GoldenPatternWithStat>& gps)
    : EmulationObserverBase(edmCfg, omtfConfig), goldenPatterns(gps) {
  // TODO Auto-generated constructor stub

  simMuPt = new TH1I("simMuPt", "simMuPt", goldenPatterns.size(), -0.5, goldenPatterns.size() - 0.5);
  simMuFoundByOmtfPt =
      new TH1I("simMuFoundByOmtfPt", "simMuFoundByOmtfPt", goldenPatterns.size(), -0.5, goldenPatterns.size() - 0.5);

  simMuPtSpectrum = new TH1F("simMuPtSpectrum", "simMuPtSpectrum", 800, 0, 400);

  simMuPtVsDispl = new TH2I("simMuPtVsDispl", "simMuPtVsDispl;pt [GeV];dxy [cm]", 100, 0, 400, 100, 0, 400);
  simMuPtVsRho = new TH2I("simMuPtVsRho", "simMuPtVsRho;pt [GeV];rho [cm]", 100, 0, 400, 100, 0, 400);

  if (edmCfg.exists("simTracksTag") == false)
    edm::LogError("l1tOmtfEventPrint") << "simTracksTag not found !!!" << std::endl;
}

PatternOptimizerBase::~PatternOptimizerBase() {}

void PatternOptimizerBase::printPatterns() {
  edm::LogVerbatim("l1tOmtfEventPrint") << __FUNCTION__ << ": " << __LINE__ << " called!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "
                                        << std::endl;
  for (int patNum = goldenPatterns.size() - 1; patNum >= 0; patNum--) {
    double pt = omtfConfig->getPatternPtRange(patNum).ptFrom;
    if (pt > 0) {
      edm::LogVerbatim("l1tOmtfEventPrint")
          << "cmsRun runThresholdCalc.py " << patNum << " " << (patNum + 1) << " _" << RPCConst::iptFromPt(pt) << "_";
      if (goldenPatterns[patNum]->key().theCharge == -1)
        edm::LogVerbatim("l1tOmtfEventPrint") << "m_";
      else
        edm::LogVerbatim("l1tOmtfEventPrint") << "p_";

      edm::LogVerbatim("l1tOmtfEventPrint") << " > out" << patNum << ".txt" << std::endl;
    }
  }
}

void PatternOptimizerBase::observeEventEnd(const edm::Event& iEvent,
                                           std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) {
  if (simMuon == nullptr || omtfCand->getGoldenPatern() == nullptr)  //no sim muon or empty candidate
    return;

  double ptSim = simMuon->momentum().pt();
  int chargeSim = (std::abs(simMuon->type()) == 13) ? simMuon->type() / -13 : 0;

  //double muDxy = (-1 * genMuon->vx() * genMuon->py() + genMuon->vy() * genMuon->px()) / genMuon->pt();;

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

void PatternOptimizerBase::savePatternsInRoot(std::string rootFileName) {
  gStyle->SetOptStat(111111);
  TFile outfile(rootFileName.c_str(), "RECREATE");
  edm::LogVerbatim("l1tOmtfEventPrint") << __FUNCTION__ << ": " << __LINE__ << " out fileName " << rootFileName
                                        << " outfile->GetName() " << outfile.GetName() << " writeLayerStat "
                                        << writeLayerStat << endl;

  outfile.cd();
  simMuFoundByOmtfPt->Write();

  simMuPtSpectrum->Write();
  simMuPtVsDispl->Write();
  simMuPtVsRho->Write();

  outfile.mkdir("patternsPdfs")->cd();
  outfile.mkdir("patternsPdfs/canvases");
  outfile.mkdir("patternsPdfs/canvases2");
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

  auto gpsCnt = goldenPatterns.size();
  auto layerCnt = goldenPatterns[0]->getPdf().size();
  auto refLayerCnt = goldenPatterns[0]->getPdf().size();

  vector<vector<TH2F*> > distPhiLayerRefLayer(layerCnt, vector<TH2F*>(refLayerCnt, nullptr));
  vector<vector<TH2F*> > meanDistPhiLayerRefLayer(layerCnt, vector<TH2F*>(refLayerCnt, nullptr));
  vector<vector<TH2F*> > pdfsLayerRefLayer(layerCnt, vector<TH2F*>(refLayerCnt, nullptr));

  for (unsigned int iLayer = 0; iLayer < layerCnt; ++iLayer) {
    int rangeFactor = 1;
    if (iLayer == 1 || iLayer == 3 || iLayer == 5)
      rangeFactor = 2;

    for (unsigned int iRefLayer = 0; iRefLayer < refLayerCnt; ++iRefLayer) {
      ostrName.str("");
      ostrName << "distPhi_refLayer_" << iRefLayer << "_layer_" << iLayer;
      ostrTtle.str("");
      ostrTtle << "distPhi refLayer " << iRefLayer << " layer " << iLayer;
      //edm::LogVerbatim("l1tOmtfEventPrint") <<__FUNCTION__<<": "<<__LINE__<<" creating hist "<<ostrTtle.str()<<std::endl;
      distPhiLayerRefLayer[iLayer][iRefLayer] = new TH2F(ostrName.str().c_str(),
                                                         ostrTtle.str().c_str(),
                                                         gpsCnt,
                                                         0,
                                                         gpsCnt,
                                                         omtfConfig->nPdfBins() * rangeFactor * 2,
                                                         (int)(omtfConfig->nPdfBins()) * (-rangeFactor) - 0.5,
                                                         omtfConfig->nPdfBins() * rangeFactor - 0.5);

      ostrName.str("");
      ostrName << "meanDistPhi_refLayer_" << iRefLayer << "_Layer_" << iLayer;
      ostrTtle.str("");
      ostrTtle << "meanDistPhi refLayer " << iRefLayer << " Layer " << iLayer;
      //edm::LogVerbatim("l1tOmtfEventPrint") <<__FUNCTION__<<": "<<__LINE__<<" creating hist "<<ostrTtle.str()<<std::endl;
      meanDistPhiLayerRefLayer[iLayer][iRefLayer] = new TH2F(ostrName.str().c_str(),
                                                             ostrTtle.str().c_str(),
                                                             gpsCnt,
                                                             0,
                                                             gpsCnt,
                                                             omtfConfig->nPdfBins() * rangeFactor * 2,
                                                             (int)(omtfConfig->nPdfBins()) * (-rangeFactor) - 0.5,
                                                             omtfConfig->nPdfBins() * rangeFactor - 0.5);

      ostrName.str("");
      ostrName << "pdfs_refLayer_" << iRefLayer << "_layer_" << iLayer;
      ostrTtle.str("");
      ostrTtle << "pdfs refLayer " << iRefLayer << " layer " << iLayer;
      //edm::LogVerbatim("l1tOmtfEventPrint") <<__FUNCTION__<<": "<<__LINE__<<" creating hist "<<ostrTtle.str()<<std::endl;
      pdfsLayerRefLayer[iLayer][iRefLayer] = new TH2F(ostrName.str().c_str(),
                                                      ostrTtle.str().c_str(),
                                                      gpsCnt,
                                                      0,
                                                      gpsCnt,
                                                      omtfConfig->nPdfBins(),
                                                      -0.5,
                                                      omtfConfig->nPdfBins() - 0.5);
    }
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

    for (unsigned int iRefLayer = 0; iRefLayer < gp->getPdf()[0].size(); ++iRefLayer) {
      for (unsigned int iLayer = 0; iLayer < gp->getPdf().size(); ++iLayer) {
        canvas->cd(1 + iLayer + iRefLayer * gp->getPdf().size());
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

        int pdfMiddle = gp->getPdf()[iLayer][iRefLayer].size() / 2;
        int shift = gp->getDistPhiBitShift(iLayer, iRefLayer);

        for (unsigned int iPdf = 0; iPdf < gp->getPdf()[iLayer][iRefLayer].size(); iPdf++) {
          hist->Fill(iPdf, gp->pdfAllRef[iLayer][iRefLayer][iPdf]);

          distPhiLayerRefLayer[iLayer][iRefLayer]->Fill(
              gp->key().theNumber,
              (((int)iPdf - pdfMiddle) << shift) + gp->meanDistPhi[iLayer][iRefLayer][0],
              gp->pdfAllRef[iLayer][iRefLayer][iPdf]);

          pdfsLayerRefLayer[iLayer][iRefLayer]->Fill(
              gp->key().theNumber, (int)iPdf, gp->pdfAllRef[iLayer][iRefLayer][iPdf]);
        }
        if ((int)iLayer == (omtfConfig->getRefToLogicNumber()[iRefLayer]))
          hist->SetLineColor(kGreen);

        meanDistPhiLayerRefLayer[iLayer][iRefLayer]->Fill(
            gp->key().theNumber, gp->meanDistPhi[iLayer][iRefLayer][0], 1);

        hist->GetYaxis()->SetRangeUser(0, omtfConfig->pdfMaxValue() + 1);
        hist->Draw("hist");

        outfile.cd("patternsPdfs");
        hist->Write();

        /////////////////////// histLayerStat
        if (writeLayerStat) {
          bool saveTh2 = false;
          if (gp->getStatistics()[iLayer][iRefLayer][0].size() > 1)
            saveTh2 = true;

          outfile.cd("layerStats");

          string histName = "histLayerStat_" + ostrName.str();
          unsigned int binCnt1 = gp->getStatistics()[iLayer][iRefLayer].size();
          if (!saveTh2) {
            TH1I* histLayerStat = new TH1I(histName.c_str(), histName.c_str(), binCnt1, -0.5, binCnt1 - 0.5);
            for (unsigned int iBin = 0; iBin < binCnt1; iBin++) {
              histLayerStat->Fill(iBin, gp->getStatistics()[iLayer][iRefLayer][iBin][0]);
            }
            histLayerStat->Write();
          } else {
            if (iRefLayer == 0 || iRefLayer == 2) {  //TODO!!!!!!!!!!!!!!!!!!!!!!!!
              unsigned int binCnt2 = gp->getStatistics()[iLayer][iRefLayer][0].size();
              //TH2I* histLayerStat = new TH2I(histName.c_str(), (histName  + ";ref phiB;delta_phi").c_str(), binCnt2, -0.5, binCnt2 - 0.5, binCnt1, -0.5, binCnt1 - 0.5);
              double xmin = -0.5 - binCnt2 / 2;
              double xmax = binCnt2 / 2 - 0.5;
              if (edmCfg.getParameter<string>("patternGenerator") == "deltaPhiVsPhiRef") {
                xmin = -0.5;
                xmax = binCnt2 - 0.5;
              }
              TH2I* histLayerStat = new TH2I(histName.c_str(),
                                             (histName + ";ref phiB;delta_phi").c_str(),
                                             binCnt2,
                                             xmin,
                                             xmax,
                                             binCnt1,
                                             -0.5 - binCnt1 / 2,
                                             binCnt1 / 2 - 0.5);

              if (edmCfg.getParameter<string>("patternGenerator") == "deltaPhiVsPhiRef")
                histLayerStat->GetXaxis()->SetTitle("ref phi");

              for (unsigned int iBin1 = 0; iBin1 < binCnt1; iBin1++) {    //deltaPhi
                for (unsigned int iBin2 = 0; iBin2 < binCnt2; iBin2++) {  //phiB
                  //histLayerStat->Fill(iBin2, iBin1, gp->getStatistics()[iLayer][iRefLayer][iBin1][iBin2]); //looks that using Fill leads to huge memory cosumption
                  histLayerStat->SetBinContent(
                      iBin2 + 1, iBin1 + 1, gp->getStatistics()[iLayer][iRefLayer][iBin1][iBin2]);
                }
              }
              histLayerStat->Write();
              histLayerStat->Delete();
            }
          }
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

  outfile.cd();
  for (auto& classProbHist : classProbHists) {
    classProbHist->Write();
  }

  outfile.cd("patternsPdfs/canvases2");
  for (unsigned int iRefLayer = 0; iRefLayer < goldenPatterns[0]->getPdf()[0].size(); ++iRefLayer) {
    ostrName.str("");
    ostrName << "distPhiForPatterns_Reflayer_" << iRefLayer;
    ostrTtle.str("");
    ostrTtle << "distPhiForPatterns Reflayer " << iRefLayer;

    TCanvas* canvas = new TCanvas(ostrName.str().c_str(), ostrTtle.str().c_str(), 1200, 1000);
    canvas->Divide(6, 3, 0, 0);

    for (unsigned int iLayer = 0; iLayer < distPhiLayerRefLayer.size(); ++iLayer) {
      canvas->cd(iLayer + 1);
      canvas->cd(iLayer + 1)->SetGridx();
      canvas->cd(iLayer + 1)->SetGridy();

      //distPhiLayerRefLayer[iLayer][iRefLayer]->SetLineColor(kBlack);
      distPhiLayerRefLayer[iLayer][iRefLayer]->Draw("colz");

      meanDistPhiLayerRefLayer[iLayer][iRefLayer]->SetLineColor(kRed);
      meanDistPhiLayerRefLayer[iLayer][iRefLayer]->Draw("boxsame");

      distPhiLayerRefLayer[iLayer][iRefLayer]->Write();
      meanDistPhiLayerRefLayer[iLayer][iRefLayer]->Write();
      pdfsLayerRefLayer[iLayer][iRefLayer]->Write();
    }

    canvas->Write();

    ostrName.str("");
    ostrName << "pdfsForPatterns_Reflayer_" << iRefLayer;
    ostrTtle.str("");
    ostrTtle << "pdfsForPatterns Reflayer " << iRefLayer;

    canvas = new TCanvas(ostrName.str().c_str(), ostrTtle.str().c_str(), 1200, 1000);
    canvas->Divide(6, 3, 0, 0);
    for (unsigned int iLayer = 0; iLayer < distPhiLayerRefLayer.size(); ++iLayer) {
      canvas->cd(iLayer + 1);
      canvas->cd(iLayer + 1)->SetGridx();
      canvas->cd(iLayer + 1)->SetGridy();

      pdfsLayerRefLayer[iLayer][iRefLayer]->Draw("colz");
    }

    canvas->Write();
  }

  saveHists(outfile);

  outfile.Close();
}
