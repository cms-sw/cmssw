/*
 * PatternOptimizerBase.cc
 *
 *  Created on: Oct 17, 2018
 *      Author: kbunkow
 */

#include <L1Trigger/L1TMuonBayes/interface/Omtf/GoldenPatternWithStat.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/XMLConfigWriter.h>
#include <L1Trigger/L1TMuonBayes/interface/OmtfPatternGeneration/PatternOptimizerBase.h>
#include "L1Trigger/RPCTrigger/interface/RPCConst.h"

#include "Math/VectorUtil.h"

#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TStyle.h"
#include <fstream>
#include "TTree.h"

PatternOptimizerBase::PatternOptimizerBase(const edm::ParameterSet& edmCfg, const OMTFConfiguration* omtfConfig, std::vector<std::shared_ptr<GoldenPatternWithStat> >& gps):
  edmCfg(edmCfg), omtfConfig(omtfConfig), goldenPatterns(gps), simMuon(0) {
  // TODO Auto-generated constructor stub

  simMuPt =  new TH1I("simMuPt", "simMuPt", goldenPatterns.size(), -0.5, goldenPatterns.size()-0.5);
  simMuFoundByOmtfPt =  new TH1I("simMuFoundByOmtfPt", "simMuFoundByOmtfPt", goldenPatterns.size(), -0.5, goldenPatterns.size()-0.5);

  simMuPtSpectrum = new TH1F("simMuPtSpectrum", "simMuPtSpectrum", 400, 0, 400);
}

PatternOptimizerBase::~PatternOptimizerBase() {
  // TODO Auto-generated destructor stub
}

void PatternOptimizerBase::printPatterns() {
  cout<<__FUNCTION__<<": "<<__LINE__<<" called!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "<<std::endl;
  for(int patNum = goldenPatterns.size() -1; patNum >= 0; patNum--  ) {
    double pt = omtfConfig->getPatternPtRange(patNum).ptFrom;
    if(pt > 0) {
      cout<<"cmsRun runThresholdCalc.py "<<patNum<<" "<<(patNum+1)<<" _"<<RPCConst::iptFromPt(pt)<<"_";
      if(goldenPatterns[patNum]->key().theCharge == -1)
        cout<<"m_";
      else
        cout<<"p_";

      cout<<" > out"<<patNum<<".txt"<<std::endl;
    }
  }
}

void PatternOptimizerBase::observeProcesorEmulation(unsigned int iProcessor, l1t::tftype mtfType,  const OMTFinput &input,
    const AlgoMuons& algoCandidates,
    const AlgoMuons& gbCandidates,
    const std::vector<l1t::RegionalMuonCand> & candMuons) {

  unsigned int procIndx = omtfConfig->getProcIndx(iProcessor, mtfType);

/*
  double ptSim = simMuon->momentum().pt();
  int chargeSim = (abs(simMuon->type()) == 13) ? simMuon->type()/-13 : 0;
  int patNum = omtfConfig->getPatternNum(ptSim, chargeSim);
  GoldenPatternWithStat* exptCandGp = goldenPatterns.at(patNum).get(); // expected pattern
*/

  //bool found = false;
  unsigned int i = 0;
  for(auto& gbCandidate : gbCandidates) {
    //int iRefHit = gbCandidate.getRefHitNumber();
    if(gbCandidate->getGoldenPatern() != 0 &&  gbCandidate->getGpResult().getFiredLayerCnt() > omtfCand->getGpResult().getFiredLayerCnt() ) {
      //cout<<__FUNCTION__<<":"<<__LINE__<<" gbCandidate "<<gbCandidate<<" "<<std::endl;
      omtfCand = gbCandidate;
      //omtfResult = gbCandidate.getGoldenPatern()->getResults()[procIndx][iRefHit]; //TODO be carrefful, because in principle the results sored by the goldenPattern can be altered in one event. In phae I omtf this should not happened, but in OMTFProcessorTTMerger - yes
      //exptResult = exptCandGp->getResults()[procIndx][iRefHit];
      candProcIndx = procIndx;

      regionalMuonCand = candMuons[i]; //not necessary good, but needed only for debug printout
      //found = true;
    }
    i++;
  }

  //////////////////////debug printout/////////////////////////////
  /*if(found) {
    GoldenPatternWithStat* omtfCandGp = static_cast<GoldenPatternWithStat*>(omtfCand.getGoldenPatern());
    if( omtfCandGp->key().thePt > 100 && exptCandGp->key().thePt <= 15 ) {
      //std::cout<<iEvent.id()<<std::endl;
      cout<<" ptSim "<<ptSim<<" chargeSim "<<chargeSim<<std::endl;
      std::cout<<"iProcessor "<<iProcessor<<" exptCandGp "<<exptCandGp->key()<<std::endl;
      std::cout<<"iProcessor "<<iProcessor<<" omtfCandGp "<<omtfCandGp->key()<<std::endl;
      std::cout<<"omtfResult "<<std::endl<<omtfResult<<std::endl;
      int refHitNum = omtfCand.getRefHitNumber();
      std::cout<<"other gps results"<<endl;
      for(auto& gp : goldenPatterns) {
        if(omtfResult.getFiredLayerCnt() == gp->getResults()[procIndx][iRefHit].getFiredLayerCnt() )
        {
          cout<<gp->key()<<std::endl<<gp->getResults()[procIndx][iRefHit]<<std::endl;
        }
      }
      std::cout<<std::endl;
    }
  }*/
}

void PatternOptimizerBase::observeEventBegin(const edm::Event& iEvent) {
  omtfCand.reset(new AlgoMuon() );
  candProcIndx = 0xffff;
  //exptResult =  GoldenPatternResult();

  simMuon = findSimMuon(iEvent);
  //cout<<__FUNCTION__<<":"<<__LINE__<<" evevt "<<iEvent.id().event()<<" simMuon pt "<<simMuon->momentum().pt()<<" GeV "<<std::endl;
}

void PatternOptimizerBase::observeEventEnd(const edm::Event& iEvent) {
  if(simMuon == 0 || omtfCand->getGoldenPatern() == 0)//no sim muon or empty candidate
    return;

  double ptSim = simMuon->momentum().pt();
  int chargeSim = (abs(simMuon->type()) == 13) ? simMuon->type()/-13 : 0;

  unsigned int exptPatNum = omtfConfig->getPatternNum(ptSim, chargeSim);
  GoldenPatternWithStat* exptCandGp = goldenPatterns.at(exptPatNum).get(); // expected pattern
  simMuFoundByOmtfPt->Fill(exptCandGp->key().theNumber); //TODO add weight of the muons pt spectrum

  simMuPtSpectrum->Fill(ptSim, getEventRateWeight(ptSim));
}

void PatternOptimizerBase::endJob() {
  std::string fName = edmCfg.getParameter<std::string>("optimisedPatsXmlFile");
  edm::LogImportant("PatternOptimizer") << " Writing optimized patterns to "<<fName << std::endl;
  XMLConfigWriter xmlWriter(omtfConfig, true, false);
  xmlWriter.writeGPs(goldenPatterns, fName);

  fName.replace(fName.find('.'), fName.length(), ".root");
  savePatternsInRoot(fName);
}

const SimTrack* PatternOptimizerBase::findSimMuon(const edm::Event &event, const SimTrack * previous) {
  const SimTrack* result = 0;
  if(edmCfg.exists("g4SimTrackSrc") == false)
    return result;

  edm::Handle<edm::SimTrackContainer> simTks;
  event.getByLabel(edmCfg.getParameter<edm::InputTag>("g4SimTrackSrc"), simTks);

  for (std::vector<SimTrack>::const_iterator it=simTks->begin(); it< simTks->end(); it++) {
    const SimTrack& aTrack = *it;
    if ( !(aTrack.type() == 13 || aTrack.type() == -13) )
      continue;
    if(previous && ROOT::Math::VectorUtil::DeltaR(aTrack.momentum(), previous->momentum()) < 0.07)
      continue;
    if ( !result || aTrack.momentum().pt() > result->momentum().pt())
      result = &aTrack;
  }
  return result;
}

void PatternOptimizerBase::savePatternsInRoot(std::string rootFileName) {
  gStyle->SetOptStat(111111);
  TFile outfile(rootFileName.c_str(), "RECREATE");
  cout<<__FUNCTION__<<": "<<__LINE__<<" out fileName "<<rootFileName<<" outfile->GetName() "<<outfile.GetName()<<endl;

  outfile.cd();
  simMuFoundByOmtfPt->Write();

  simMuPtSpectrum->Write();

  outfile.mkdir("patternsPdfs")->cd();
  outfile.mkdir("patternsPdfs/canvases");
  ostringstream ostrName;
  ostringstream ostrTtle;
  vector<TH1F*> classProbHists;
  for(unsigned int iRefLayer = 0; iRefLayer < goldenPatterns[0]->getPdf()[0].size(); ++iRefLayer) {
    ostrName.str("");
    ostrName<<"Neg_RefLayer_"<<iRefLayer;
    ostrTtle.str("");
    ostrTtle<<"Neg_RefLayer_"<<iRefLayer;
    classProbHists.push_back(new TH1F(ostrName.str().c_str(), ostrTtle.str().c_str(), goldenPatterns.size(), -0.5, goldenPatterns.size() - 0.5) );

    ostrName.str("");
    ostrName<<"Pos_RefLayer_"<<iRefLayer;
    ostrTtle.str("");
    ostrTtle<<"Pos_RefLayer_"<<iRefLayer;
    classProbHists.push_back(new TH1F(ostrName.str().c_str(), ostrTtle.str().c_str(), goldenPatterns.size(), -0.5, goldenPatterns.size() - 0.5) );
  }

  for(auto& gp : goldenPatterns) {
    OMTFConfiguration::PatternPt patternPt = omtfConfig->getPatternPtRange(gp->key().theNumber);
    if(gp->key().thePt == 0)
      continue;
    //cout<<__FUNCTION__<<": "<<__LINE__<<" "<<gp->key()<<std::endl;
    ostrName.str("");
    ostrName<<"PatNum_"<<gp->key().theNumber;
    ostrTtle.str("");
    ostrTtle<<"PatNum_"<<gp->key().theNumber<<"_ptCode_"<<gp->key().thePt<<"_Pt_"<<patternPt.ptFrom<<"_"<<patternPt.ptTo<<"_GeV";
    TCanvas* canvas = new TCanvas(ostrName.str().c_str(), ostrTtle.str().c_str(), 1200, 1000);
    canvas->Divide(gp->getPdf().size(), gp->getPdf()[0].size(), 0, 0);
    outfile.cd("patternsPdfs");
    for(unsigned int iLayer = 0; iLayer < gp->getPdf().size(); ++iLayer) {
      for(unsigned int iRefLayer = 0; iRefLayer < gp->getPdf()[iLayer].size(); ++iRefLayer) {
        canvas->cd(1 + iLayer + iRefLayer * gp->getPdf().size());
        //unsigned int refLayerLogicNumber = omtfConfig->getRefToLogicNumber()[iRefLayer];
        ostrName.str("");
        ostrName<<"PatNum_"<<gp->key().theNumber<<"_refLayer_"<<iRefLayer<<"_Layer_"<<iLayer;
        ostrTtle.str("");
        ostrTtle<<"PatNum "<<gp->key().theNumber<<" ptCode "<<gp->key().thePt<<" refLayer "<<iRefLayer<<" Layer "<<iLayer
            <<" meanDistPhi "<<gp->meanDistPhi[iLayer][iRefLayer][0]<<" distPhiBitShift "<<gp->getDistPhiBitShift(iLayer, iRefLayer); //"_Pt_"<<patternPt.ptFrom<<"_"<<patternPt.ptTo<<"_GeV
        //cout<<__FUNCTION__<<": "<<__LINE__<<" creating hist "<<ostrTtle.str()<<std::endl;
        TH1F* hist = new TH1F(ostrName.str().c_str(), ostrTtle.str().c_str(), omtfConfig->nPdfBins(), -0.5, omtfConfig->nPdfBins()-0.5);
        for(unsigned int iPdf = 0; iPdf < gp->getPdf()[iLayer][iRefLayer].size(); iPdf++) {
          hist->Fill(iPdf, gp->pdfAllRef[iLayer][iRefLayer][iPdf]);
        }
        if((int)iLayer == (omtfConfig->getRefToLogicNumber()[iRefLayer]))
          hist->SetLineColor(kGreen);
        hist->Write();
        hist->Draw("hist");
      }
    }
    outfile.cd("patternsPdfs/canvases");
    canvas->Write();
    delete canvas;

    unsigned int iPdf = omtfConfig->nPdfBins()/2;
    for(unsigned int iRefLayer = 0; iRefLayer < gp->getPdf()[0].size(); ++iRefLayer) {
      unsigned int refLayerLogicNumber = omtfConfig->getRefToLogicNumber()[iRefLayer];
      if(gp->key().theCharge == -1) {
        classProbHists[2 * iRefLayer]->Fill(gp->key().theNumber, gp->pdfAllRef[refLayerLogicNumber][iRefLayer][iPdf]);
      }
      else
        classProbHists[2 * iRefLayer +1]->Fill(gp->key().theNumber, gp->pdfAllRef[refLayerLogicNumber][iRefLayer][iPdf]);
    }
  }

  outfile.mkdir("patternsPdfSumStat")->cd();
  for(auto& gp : goldenPatterns) {
    for(unsigned int iRefLayer = 0; iRefLayer < gp->gpProbabilityStat.size(); ++iRefLayer) {
      gp->gpProbabilityStat[iRefLayer]->Write();
    }
  }

  outfile.cd();
  for(auto& classProbHist : classProbHists) {
    classProbHist->Write();
  }

  saveHists(outfile);

  outfile.Close();
}
