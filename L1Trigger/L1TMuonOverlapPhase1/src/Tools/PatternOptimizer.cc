/*
 * PatternOptimizer.cc
 *
 *  Created on: Oct 12, 2017
 *      Author: kbunkow
 */

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/PatternOptimizer.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/AlgoMuon.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPattern.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternResult.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/StubResult.h"
#include "Math/GenVector/LorentzVector.h"
#include "SimDataFormats/Track/interface/CoreSimTrack.h"
#include "SimDataFormats/Track/interface/SimTrack.h"

#include "TAxis.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"

#include <boost/range/adaptor/reversed.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "boost/multi_array/multi_array_ref.hpp"
#include "boost/multi_array/subarray.hpp"

PatternOptimizer::PatternOptimizer(const edm::ParameterSet& edmCfg,
                                   const OMTFConfiguration* omtfConfig,
                                   GoldenPatternVec<GoldenPatternWithStat>& gps)
    : PatternOptimizerBase(edmCfg, omtfConfig, gps),
      //TODO set desire function here, see https://www.cprogramming.com/c++11/c++11-lambda-closures.html
      updateStatFunc([this](GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) {
        updateStatCollectProb(omtfCandGp, exptCandGp);
      }),
      //updateStatFunc([this] (GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) { updateStatForAllGps(omtfCandGp, exptCandGp); } ),
      //updateStatFunc([this] (GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) { calculateThresholds(omtfCandGp, exptCandGp); } ),
      //updateStatFunc([this] (GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) { tuneClassProb(omtfCandGp, exptCandGp); } ),
      //updateStatFunc([this] (GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) { updateStatCloseResults(omtfCandGp, exptCandGp); } ),

      //updateStatFunc([this] (GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) { updateStatVoter_1(omtfCandGp, exptCandGp); } ),
      //updateStatFunc([this] (GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) { updateStatPtDiff_1(omtfCandGp, exptCandGp); } ),
      //updateStatFunc([this] (GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) { updateStatPtDiff2_1(omtfCandGp, exptCandGp); } ),
      //updateStatFunc([this] (GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) { updateStatPtLogDiff_2(omtfCandGp, exptCandGp); } ),
      //updatePdfsFunc([this] (GoldenPatternWithStat* gp, unsigned int& iLayer, unsigned int& iRefLayer, double& learingRate) { updatePdfsMean_1(gp, iLayer, iRefLayer, learingRate); })
      //updatePdfsFunc([this] (GoldenPatternWithStat* gp, unsigned int& iLayer, unsigned int& iRefLayer, double& learingRate) { updatePdfsMean_2(gp, iLayer, iRefLayer, learingRate); })
      //updatePdfsFunc([this] (GoldenPatternWithStat* gp, unsigned int& iLayer, unsigned int& iRefLayer, double& learingRate) { updatePdfsVoter_1(gp, iLayer, iRefLayer, learingRate); })
      updatePdfsFunc(
          [this](GoldenPatternWithStat* gp, unsigned int& iLayer, unsigned int& iRefLayer, double& learingRate) {
            updatePdfsAnaDeriv(gp, iLayer, iRefLayer, learingRate);
          })
//updatePdfsFunc([this] (GoldenPatternWithStat* gp, unsigned int& iLayer, unsigned int& iRefLayer, double& learingRate) { updatePdfsNumDeriv(gp, iLayer, iRefLayer, learingRate); })
{
  //modifyPatterns(); //TODO remove if not needed!!!!!!!!!!!!!!!!

  //ptRangeFrom = edmCfg.getParameter<double>("ptRangeFrom");
  //ptRangeFrom = edmCfg.getParameter<double>("ptRangeTo");

  if (edmCfg.exists("selectedPatNum"))
    selectedPatNum = edmCfg.getParameter<unsigned int>("selectedPatNum");  //TODO
  else
    selectedPatNum = -1;

  if (edmCfg.exists("deltaPdf"))
    deltaPdf = edmCfg.getParameter<double>("deltaPdf");
  else
    deltaPdf = 0;

  //ptCut = goldenPatterns[selectedPatNum]->key().thePt;
  optPatXmlFile = edmCfg.getParameter<string>("optimisedPatsXmlFile");

  //currnetPtBatchPatNum = selectedPatNum;

  //printPatterns();

  initRateWeights();

  //modifyPatterns(); //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  selRefL = 0;
  selL1 = 1;
  selL2 = 2;
  deltaPhi1_deltaPhi2_hits.assign(goldenPatterns.size(), nullptr);
  deltaPhi1_deltaPhi2_omtf.assign(goldenPatterns.size(), nullptr);

  ostringstream ostrName, ostrTtle;
  for (auto& gp : goldenPatterns) {
    if (gp->key().thePt == 0)
      continue;

    gp->iniStatisitics(gp->pdfAllRef[0][0].size(), 4);  //TODO

    gp->initGpProbabilityStat();

    ostrName.str("");
    ostrName << "PatNum_" << gp->key().theNumber;
    ostrTtle.str("");
    ostrTtle << "PatNum_" << gp->key().theNumber << "_ptCode_"
             << gp->key().thePt;  //<<"_Pt_"<<patternPt.ptFrom<<"_"<<patternPt.ptTo<<"_GeV";

    int nBins = omtfConfig->nPdfBins();
    float from = -0.5;
    float to = omtfConfig->nPdfBins() - 0.5;
    deltaPhi1_deltaPhi2_hits[gp->key().theNumber] = new TH2I(
        (ostrName.str() + "_hits").c_str(), (ostrTtle.str() + "_hits").c_str(), nBins, from, to, nBins, from, to);
    deltaPhi1_deltaPhi2_omtf[gp->key().theNumber] = new TH2I(
        (ostrName.str() + "_omtf").c_str(), (ostrTtle.str() + "_omtf").c_str(), nBins, from, to, nBins, from, to);
  }
}

//suppressing tails of the distributions
void PatternOptimizer::modifyPatterns() {
  cout << __FUNCTION__ << ": " << __LINE__ << " called!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << std::endl;
  for (auto& gp : goldenPatterns) {
    if (gp->key().thePt == 0)
      continue;
    for (unsigned int iLayer = 0; iLayer < gp->getPdf().size(); ++iLayer) {
      for (unsigned int iRefLayer = 0; iRefLayer < gp->getPdf()[iLayer].size(); ++iRefLayer) {
        unsigned int refLayerLogicNumber = omtfConfig->getRefToLogicNumber()[iRefLayer];
        if (refLayerLogicNumber == iLayer)  //dont do the pdf cleaning for the P(Ck)
          continue;
        for (unsigned int iPdf = 1; iPdf < gp->getPdf()[iLayer][iRefLayer].size(); iPdf++) {
          /*          if( gp->pdfAllRef[iLayer][iRefLayer][iPdf] &&
              gp->pdfAllRef[iLayer][iRefLayer][iPdf -1] == 0 &&  //neigbours must be 0
              (iPdf != gp->getPdf()[iLayer][iRefLayer].size()-1 && gp->pdfAllRef[iLayer][iRefLayer][iPdf +1] == 0 ) &&
             (   (gp->key().thePt > 30  && gp->pdfAllRef[iLayer][iRefLayer][iPdf] < 0.005)
              || (gp->key().thePt <= 30 && gp->pdfAllRef[iLayer][iRefLayer][iPdf] < 0.001) )) {//suppressing tails of the distributions
            cout<<__FUNCTION__<<": "<<__LINE__<<" "<<gp->key()<<" pdf bin: iLayer "<<iLayer<<" iRefLayer "<<iRefLayer<<" iPdf "<<iPdf
                <<" val "<<gp->pdfAllRef[iLayer][iRefLayer][iPdf]<<" set to 0"<<std::endl;
            gp->pdfAllRef[iLayer][iRefLayer][iPdf] = 0;
          }*/

          if (gp->pdfAllRef[iLayer][iRefLayer][iPdf] != 0 &&
              gp->pdfAllRef[iLayer][iRefLayer][iPdf] < 0.001) {  //suppressing tails of the distributions
            cout << __FUNCTION__ << ": " << __LINE__ << " " << gp->key() << " pdf bin: iLayer " << iLayer
                 << " iRefLayer " << iRefLayer << " iPdf " << iPdf << " val " << gp->pdfAllRef[iLayer][iRefLayer][iPdf]
                 << " set to 0" << std::endl;
            gp->pdfAllRef[iLayer][iRefLayer][iPdf] = 0;
          }
        }
      }
    }
  }
}

PatternOptimizer::~PatternOptimizer() {}

void PatternOptimizer::observeEventEnd(const edm::Event& iEvent,
                                       std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) {
  if (simMuon == nullptr || omtfCand->getGoldenPatern() == nullptr)  //no sim muon or empty candidate
    return;

  //cout<<__FUNCTION__<<":"<<__LINE__<<" event "<<iEvent.id().event()<<endl;
  //cout<<__FUNCTION__<<":"<<__LINE__<<" omtfCand "<<omtfCand<<std::endl;
  //cout<<__FUNCTION__<<":"<<__LINE__<<" omtfCand->getGpResult() "<<std::endl<<omtfCand->getGpResult()<<std::endl;
  PatternOptimizerBase::observeEventEnd(iEvent, finalCandidates);

  GoldenPatternWithStat* omtfCandGp = static_cast<GoldenPatternWithStat*>(omtfCand->getGoldenPatern());

  double ptSim = simMuon->momentum().pt();
  int chargeSim = (abs(simMuon->type()) == 13) ? simMuon->type() / -13 : 0;

  unsigned int exptPatNum = omtfConfig->getPatternNum(ptSim, chargeSim);
  GoldenPatternWithStat* exptCandGp = goldenPatterns.at(exptPatNum).get();  // expected pattern

  int iRefHit = omtfCand->getRefHitNumber();
  //GoldenPatternResult omtfCand->getGpResult() = omtfCandGp->getResults()[candProcIndx][iRefHit];
  exptResult = exptCandGp->getResults()[candProcIndx][iRefHit];

  updateStatFunc(omtfCandGp, exptCandGp);

  ///debug printout
  //  if( omtfCandGp->key().thePt > 40 && exptCandGp->key().thePt <= 15 )
  //  {
  //    std::cout<<iEvent.id()<<std::endl;
  //    std::cout<<" ptSim "<<ptSim<<" chargeSim "<<chargeSim<<" patNum "<<exptPatNum<<std::endl;
  //    std::cout<<"exptCandGp "<<exptCandGp->key()<<std::endl;
  //    std::cout<<"exptResult "<<std::endl<<exptResult<<std::endl;
  //
  //    std::cout<<"omtfCandGp "<<omtfCandGp->key()<<std::endl;
  //    std::cout<<"omtfCand->getGpResult() "<<std::endl<<omtfCand->getGpResult()<<std::endl;
  //    /*std::cout<<"other gps results"<<endl;
  //    for(auto& gp : goldenPatterns) {
  //      if(omtfCand->getGpResult().getFiredLayerCnt() == gp->getResults()[candProcIndx][iRefHit].getFiredLayerCnt() )
  //      {
  //        cout<<gp->key()<<std::endl<<gp->getResults()[candProcIndx][iRefHit]<<std::endl;
  //      }
  //    }*/
  //    std::cout<<std::endl;
  //  }
}

void PatternOptimizer::endJob() {
  //savePatternsInRoot("orginalPatterns.root");

  //TODO select the function to be executed
  //updatePdfCloseResults();

  //calulateProb();
  //tuneClassProb(0.8);
  /*  double targetEff = edmCfg.getParameter<double>("targetEff");
  calculateThresholds(targetEff);*/

  /*  double learingRate = 0.01;
  for(auto& gp : goldenPatterns) {
    for(unsigned int iLayer = 0; iLayer < gp->getPdf().size(); ++iLayer) {
      for(unsigned int iRefLayer = 0; iRefLayer < gp->getPdf()[iLayer].size(); ++iRefLayer) {
        updatePdfsFunc(gp.get(), iLayer, iRefLayer, learingRate);
      }
    }
  }*/

  /*
  double step = edmCfg.getParameter<double>("step");
  modifyPatterns1(step); //TODO remove!!!!!!!!!!!!!!!!!!!!!!!!
*/

  //calculateThresholds
  /*  for(int i = 0; i <= 10; i++) {
    //TODO select what to do
    double targetEff = 0.70 + 0.02 * i;
    calculateThresholds(targetEff);

    //double targetEff = edmCfg.getParameter<double>("targetEff"); //0.01 * i; //targetEffLost
    //tuneClassProb(targetEff); //connot be done in the loop, because the pdf's are updated each time, to make it good the original gp should be copied

    std::ostringstream fName;// = edmCfg.getParameter<std::string>("optimisedPatsXmlFile");

    fName<<"optimisedPats_thr_0"<<setw(4)<<setfill('0')<<(targetEff * 100)<<".xml";
    //fName<<"optimisedPats_"<<selectedPatNum<<".xml";

    edm::LogImportant("PatternOptimizer") << " Writing  patterns with thresholds to "<<fName.str() << std::endl;
    //XMLConfigWriter xmlWriter(omtfConfig, true, false);
    xmlWriter.writeGPs(goldenPatterns, fName.str());
  }*/

  PatternOptimizerBase::endJob();

  std::ofstream outfile;
  outfile.open("test.txt", std::ios_base::app);
  outfile << optPatXmlFile << "\tERROR=\t" << errorSum << "\tN_OF_EVENTS=\t" << nEvents << "\tAVERAGE_ERROR=\t"
          << errorSum / 2.0 / nEvents << "";
  outfile << "\tRefL_Error=(err, N, err/N)\t";
  int i = 0;
  for (auto a : errorSumRefL) {
    outfile << i << ":\t" << a << "\t" << nEventsRefL.at(i) << "\t"
            << ((nEventsRefL.at(i) > 0) ? a / nEventsRefL.at(i) : 0) << "\t";
    i++;
  }
  outfile << std::endl;
}

void PatternOptimizer::updateStatCollectProb(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) {
  /*  for(unsigned int iLayer = 0;  iLayer < exptResult.getHitPdfBins().size(); iLayer++) {
    //updating statistic for the gp which should have fired
    int iBinExpt = exptResult.getHitPdfBins()[iLayer];
    if(iBinExpt != 0) {
      exptCandGp->updateStat(iLayer, exptResult.getRefLayer(), iBinExpt, whatExptVal, 1); //TODO in principle the events should be weighted by the muon pt probability
      exptCandGp->updateStat(iLayer, exptResult.getRefLayer(), iBinExpt, whatExptNorm, 1);
      //cout<<__FUNCTION__<<":"<<__LINE__<<" updating statistic for exptCandGp: iLayer "<<iLayer<<" RefLayer "<<exptResult.getRefLayer()<<" iBinExpt "<<iBinExpt<<" fired "<<exptResult.isLayerFired(iLayer)<<std::endl;
    }
  }*/

  if (regionalMuonCand.hwQual() >= 12 && exptCandGp->key().thePt <= 12 && omtfCandGp->key().thePt >= 30) {
    cout << __FUNCTION__ << ":" << __LINE__ << " omtfCand " << omtfCand << std::endl;
    cout << __FUNCTION__ << ":" << __LINE__ << " regionalMuonCand hwPt " << regionalMuonCand.hwPt() << " hwQual "
         << regionalMuonCand.hwQual() << std::endl;
    cout << __FUNCTION__ << ":" << __LINE__ << " " << omtfCandGp->key() << " omtfCand->getGpResult()\n"
         << omtfCand->getGpResult() << std::endl;
    cout << __FUNCTION__ << ":" << __LINE__ << " " << exptCandGp->key() << " exptResult\n" << exptResult << std::endl;
    cout << "-------------------------------------" << endl;
  }

  for (unsigned int iRefHit = 0; iRefHit < exptCandGp->getResults()[candProcIndx].size(); iRefHit++) {
    GoldenPatternResult result = exptCandGp->getResults()[candProcIndx][iRefHit];
    double eventWeight = 1;
    if (exptCandGp->key().thePt == 41)
      eventWeight = 5. / 2.;
    else if (exptCandGp->key().thePt == 45)
      eventWeight = 5. / 3.;

    for (unsigned int iLayer = 0; iLayer < result.getStubResults().size(); iLayer++) {
      //updating statistic for the gp which should have fired
      int iBinExpt = result.getStubResults()[iLayer].getPdfBin();
      if (iBinExpt != 0) {
        exptCandGp->updateStat(iLayer, result.getRefLayer(), iBinExpt, whatExptVal, 1);
        //exptCandGp->updateStat(iLayer, result.getRefLayer(), iBinExpt, whatExptNorm, rateWeights[exptPatNum]); //TODO version with events weighted by the muon pt probability (i.e. rate)
        exptCandGp->updateStat(
            iLayer,
            result.getRefLayer(),
            iBinExpt,
            whatExptNorm,
            eventWeight);  //TODO version without modifying the muon pt spectrum - it should be flat in OMTF bins
        //exptCandGp->updateStat(iLayer, result.getRefLayer(), iBinExpt, whatExptNorm, getEventRateWeight(simMuon->momentum().pt()) ); //TODO version with vxMuRate muon pt spectrum

        //cout<<__FUNCTION__<<":"<<__LINE__<<" updating statistic for exptCandGp: iLayer "<<iLayer<<" RefLayer "<<exptResult.getRefLayer()<<" iBinExpt "<<iBinExpt<<" fired "<<exptResult.isLayerFired(iLayer)<<std::endl;
      }
    }
  }

  if (exptResult.getRefLayer() == selRefL) {
    int delatPhi1 = exptResult.getStubResults()[selL1].getPdfBin();
    int delatPhi2 = exptResult.getStubResults()[selL2].getPdfBin();
    if (delatPhi1 && delatPhi2) {
      delatPhi1 += exptCandGp->meanDistPhiValue(selL1, selRefL);
      delatPhi2 += exptCandGp->meanDistPhiValue(selL2, selRefL);
      deltaPhi1_deltaPhi2_hits[exptCandGp->key().theNumber]->Fill(delatPhi1, delatPhi2);
    }
  }

  if (omtfCand->getGpResult().getRefLayer() == selRefL) {
    int delatPhi1 = omtfCand->getGpResult().getStubResults()[selL1].getPdfBin();
    int delatPhi2 = omtfCand->getGpResult().getStubResults()[selL2].getPdfBin();
    if (delatPhi1 && delatPhi2) {
      delatPhi1 += omtfCandGp->meanDistPhiValue(selL1, selRefL);
      delatPhi2 += omtfCandGp->meanDistPhiValue(selL2, selRefL);
      deltaPhi1_deltaPhi2_omtf[omtfCandGp->key().theNumber]->Fill(delatPhi1, delatPhi2);
    }
  }
}

void PatternOptimizer::calulateProb() {
  for (auto& gp : goldenPatterns) {
    if (gp->key().thePt == 0)
      continue;
    cout << __FUNCTION__ << ": " << __LINE__ << " " << gp->key() << " Calculating P(x | C_k) " << std::endl;
    for (unsigned int iLayer = 0; iLayer < gp->getPdf().size(); ++iLayer) {
      for (unsigned int iRefLayer = 0; iRefLayer < gp->getPdf()[iLayer].size(); ++iRefLayer) {
        unsigned int refLayerLogicNumber = omtfConfig->getRefToLogicNumber()[iRefLayer];
        if (iLayer == refLayerLogicNumber)  //skip this as here we keep the P(C_k),
          continue;

        double pdfCnt = 0;
        double pdfNorm = 0;
        for (unsigned int iPdf = 0; iPdf < gp->getPdf()[iLayer][iRefLayer].size(); iPdf++) {
          pdfCnt += (double)gp->statistics[iLayer][iRefLayer][iPdf][whatExptVal];
          pdfNorm += (double)gp->statistics[iLayer][iRefLayer][iPdf][whatExptNorm];
        }

        cout << __FUNCTION__ << ":" << __LINE__ << " : iLayer " << iLayer << " RefLayer " << iRefLayer << " pdfNorm "
             << pdfNorm << " pdfCnt " << pdfCnt << std::endl;

        for (unsigned int iPdf = 0; iPdf < gp->getPdf()[iLayer][iRefLayer].size(); iPdf++) {
          if (pdfCnt > 500) {  //50 is to reject the one with very small statistics
            double prob = (double)gp->statistics[iLayer][iRefLayer][iPdf][whatExptNorm] / pdfNorm;
            gp->pdfAllRef[iLayer][iRefLayer][iPdf] = prob;
          } else
            gp->pdfAllRef[iLayer][iRefLayer][iPdf] = 0;
        }
      }
    }
  }

  //in pdf[32] value for the ref layer we keep the "class probability" P(C_k), where class is the pt-sign (i.e. golden pattern)
  cout << __FUNCTION__ << ": " << __LINE__ << " Calculating P(C_k) " << std::endl;
  unsigned int iPdf = omtfConfig->nPdfBins() / 2;  // <<(omtfConfig->nPdfAddrBits()-1);
  for (unsigned int iRefLayer = 0; iRefLayer < goldenPatterns[0]->getPdf()[0].size(); ++iRefLayer) {
    double cnt = 0;
    double norm = 0;
    unsigned int refLayerLogicNumber = omtfConfig->getRefToLogicNumber()[iRefLayer];
    for (auto& gp : goldenPatterns) {
      if (gp->key().thePt == 0)
        continue;
      cnt += (double)gp->statistics[refLayerLogicNumber][iRefLayer][iPdf][whatExptVal];
      norm += (double)gp->statistics[refLayerLogicNumber][iRefLayer][iPdf][whatExptNorm];
    }

    cout << __FUNCTION__ << " Calculating P(C_k) "
         << ":" << __LINE__ << " RefLayer " << iRefLayer << " norm " << norm << std::endl;
    for (auto& gp : goldenPatterns) {
      if (gp->key().thePt == 0)
        continue;
      cout << __FUNCTION__ << ": " << __LINE__ << " " << gp->key() << " Calculating P(C_k) " << std::endl;
      if (cnt > 500)
        gp->pdfAllRef[refLayerLogicNumber][iRefLayer][iPdf] =
            (double)gp->statistics[refLayerLogicNumber][iRefLayer][iPdf][whatExptNorm] / norm;
      else
        gp->pdfAllRef[refLayerLogicNumber][iRefLayer][iPdf] = 0;
    }
  }

  modifyPatterns();
}

/*void PatternOptimizer::checkNEventsInFiles()
{
  for (int pt = 1; pt <= 31; pt++)
    for (int iter = 1; iter <= 101; iter++){
      TFile *f = TFile::Open("http://lcg-heppkg.web.cern.ch/lcg-heppkg/ROOT/eventdata.root");
    }
}*/

//gradient descent optimisation of pdf-s - tu usee this the OMTF must calculate the probabilities i.e. use the sorterWithThreshold and sorterWithThresholdMode  = cms.string("bestGPByMaxGpProbability1")
void PatternOptimizer::updateStatForAllGps(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) {
  /*  std::vector<double> weights = {
      0,    //0
      30000,//1
      20000, //2
      5000, //3
      1000, //4
      500,  //5
      200,  //6
      100,  //7
      30,   //8
      10,   //9
      5,    //10
      1     //11
  };*/
  //boost::timer::auto_cpu_timer t("%ws wall, %us user in " + string(__FUNCTION__) + "\n");
  unsigned int iRefHit = omtfCand->getRefHitNumber();

  double eventWeight = 1;

  //remember that thePt is not in GeV, but 2*GeV + 1
  if (exptCandGp->key().thePt <=
      10) {  //TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! adding more files with low PT muons - must fit to the phyton
    eventWeight = 1. / 3.;
  }

  //int omtfCandPtCode = patternPtCodes[omtfCandGp->key().theNumber];
  //int exptCandPtCode = patternPtCodes[exptCandGp->key().theNumber];

  //  if(exptCandGp->key().thePt >= 41 && omtfCandGp->key().thePt < 41) {
  //    eventWeight = sqrt(patternPtCodes[exptCandGp->key().theNumber] - patternPtCodes[omtfCandGp->key().theNumber]);
  //    if(omtfCand->getRefLayer() == 1)
  //      eventWeight *= 10;
  //  }
  //  else if(exptCandGp->key().thePt < 37 && omtfCandGp->key().thePt >= 41) {
  //    //eventWeight = 6./sqrt(exptCandGp->key().thePt);
  //    //eventWeight = sqrt(patternPtCodes[omtfCandGp->key().theNumber] - patternPtCodes[exptCandGp->key().theNumber]);
  ///*    if(omtfCand->getRefLayer() == 1 || omtfCand->getRefLayer() == 4 || omtfCand->getRefLayer() == 5)
  //      eventWeight = 55./patternPtCodes[exptCandGp->key().theNumber] - 4;
  //    else if(omtfCand->getRefLayer() == 3 )
  //      eventWeight = 33./patternPtCodes[exptCandGp->key().theNumber] - 2;
  //    else if(omtfCand->getRefLayer() == 0) {
  //      eventWeight = 5500./patternPtCodes[exptCandGp->key().theNumber] - 499; //Key_41: (eta=0, pt=41, charge=1) ptCode 12; Key_40: (eta=0, pt=37, charge=1) ptCode 11
  //    }
  //    else if(omtfCand->getRefLayer() == 2) {
  //      eventWeight = 5500./patternPtCodes[exptCandGp->key().theNumber] - 499; //Key_41: (eta=0, pt=41, charge=1) ptCode 12; Key_40: (eta=0, pt=37, charge=1) ptCode 11
  //    }
  //    else {
  //      eventWeight = 110./patternPtCodes[exptCandGp->key().theNumber] - 9;
  //    }*/
  //
  //    /*if(omtfCand->getRefLayer() == 3 || omtfCand->getRefLayer() == 5)
  //      eventWeight = 110./patternPtCodes[exptCandGp->key().theNumber] - 9;*/
  //
  //    eventWeight = weights[patternPtCodes[exptCandGp->key().theNumber] ];
  //    if(omtfCand->getRefLayer() == 1 || omtfCand->getRefLayer() == 3 || omtfCand->getRefLayer() == 4 || omtfCand->getRefLayer() == 5)
  //      eventWeight *= 0.2 ;
  //    else if(omtfCand->getRefLayer() == 2)
  //      eventWeight *= 0.5 ;
  //  }

  //to handle additional pattern with pt == 45 (22GeV)
  if (exptCandGp->key().thePt == 41)
    eventWeight *= 5. / 2.;
  else if (exptCandGp->key().thePt == 45)
    eventWeight *= 5. / 3.;

  //eventWeight = 1; //TODO remove!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

  nEvents++;
  nEventsRefL.at(omtfCand->getRefLayer()) += eventWeight;

  //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<omtfCandGp->key()<<" omtfCand->getGpResult()\n"<<omtfCand->getGpResult()<<std::endl;
  //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<exptCandGp->key()<<" exptResult\n"<<exptResult<<std::endl;
  double probSum = 0;
  double errorFun = 0;
  for (auto& itGP : goldenPatterns) {
    if (itGP->key().thePt == 0)
      continue;

    auto& result = itGP->getResults()[candProcIndx][iRefHit];
    if (result.getFiredLayerCnt() < omtfCand->getGpResult().getFiredLayerCnt())
      itGP->gpProbabilityStat[result.getRefLayer()]->Fill(0);
    else {
      double gpProbability = result.getGpProbability1();
      //      cout<<__FUNCTION__<<":"<<__LINE__<<" "<<itGP->key()<<" iRefHit "<<iRefHit<<" gpProbability "<<gpProbability<<" FiredLayerCnt "<<result.getFiredLayerCnt()<<std::endl;
      itGP->gpProbabilityStat[result.getRefLayer()]->Fill(gpProbability);

      //here y is only to calculate error and corresponds to itGP
      double y = 0;
      if (itGP.get() == exptCandGp) {
        y = 1;
      } else {
        y = 0;
      }
      errorFun += pow(y - gpProbability, 2);

      double p_deltaPhi1 = result.getPdfSum() / result.getGpProbability1();
      double pdfSum = result.getPdfSum();

      if (gpProbability > 1. || gpProbability < 0) {
        cout << __FUNCTION__ << ":" << __LINE__ << " gpProbability = " << gpProbability << " err " << errorFun
             << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        cout << __FUNCTION__ << ":" << __LINE__ << " " << itGP->key() << "\n"
             << itGP->getResults()[candProcIndx][iRefHit] << "\n omtfCand->getGpResult() " << omtfCandGp->key() << "\n"
             << omtfCand->getGpResult() << "\n"
             << "exptCandGp" << exptCandGp->key() << endl
             << "---------------------------------------------" << endl;
      }

      probSum += gpProbability;
      //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<itGP->key()<<"\n"<<itGP->getResults()[candProcIndx][iRefHit]<<endl;
      //cout<<__FUNCTION__<<":"<<__LINE__<<" gpProbability = "<<gpProbability<<" y - gpProbability "<<(y - gpProbability)<<" pow "<<(pow(y - gpProbability, 2))<<endl;

      //this loop will be needed for training
      for (unsigned int iLayer = 0; iLayer < result.getStubResults().size(); iLayer++) {
        int iBin = result.getStubResults()[iLayer].getPdfBin();
        if (iBin != 0) {
          double gradErrorAna = 0;
          double gradErrorNum = 0;

          double pdfValue = result.getStubResults()[iLayer].getPdfVal();
          if (pdfValue == 0)
            continue;

          for (auto& itGPk : goldenPatterns) {
            if (itGPk->getResults()[candProcIndx][iRefHit].getFiredLayerCnt() <
                omtfCand->getGpResult().getFiredLayerCnt())
              continue;

            //here y corresponds to itGPk
            if (itGPk.get() == exptCandGp) {
              y = 1;
            } else {
              y = 0;
            }
            double gpProbabilityK = itGPk->getResults()[candProcIndx][iRefHit].getGpProbability1();
            //gpProbability is k'
            //gpProbabilityK is k

            if (itGP == itGPk)
              gradErrorAna += -2 * (y - gpProbability) * (1 - gpProbability) * gpProbability / pdfValue;
            else
              gradErrorAna += 2 * (y - gpProbabilityK) * gpProbabilityK * gpProbability / pdfValue;

            double pdfSumk = itGPk->getResults()[candProcIndx][iRefHit].getPdfSum();
            //double p_deltaPhi1k = pdfSumk / gpProbabilityK;
            gradErrorNum +=
                -(pow(y - (pdfSum * (1 + deltaPdf / pdfValue)) / (p_deltaPhi1 + pdfSumk * deltaPdf / pdfValue), 2) -
                  pow(y - gpProbability, 2)) /
                deltaPdf;
          }
          itGP->updateStat(iLayer, result.getRefLayer(), iBin, whatExptNorm, gradErrorNum * eventWeight);
          itGP->updateStat(iLayer, result.getRefLayer(), iBin, whatExptVal, gradErrorAna * eventWeight);
          //          std::cout<<__FUNCTION__<<":"<<__LINE__<<" "<<iBin<<" pdfValue "<<pdfValue
          //                <<" AnaDeriv "<<gradErrorAna
          //                <<" deltaPdf "<<deltaPdf
          //                <<" NumDeriv "<<gradErrorNum << std::endl;

          //          cout<<__FUNCTION__<<":"<<__LINE__<<" updating statistic for exptCandGp: iLayer "<<iLayer<<" RefLayer "<<exptResult.getRefLayer()<<" iBinExpt "<<iBinExpt<<" fired "<<exptResult.isLayerFired(iLayer)<<std::endl;
        }
      }
    }
  }
  errorSum += errorFun;
  errorSumRefL.at(omtfCand->getRefLayer()) += (errorFun * eventWeight);
  if (probSum > 1.0001 || probSum < 0.99) {
    cout << __FUNCTION__ << ":" << __LINE__ << " probSum " << probSum << " err " << errorFun
         << (probSum > 1 || probSum < 0.99 ? " !!!!!!!!!!!!!!!!!!!!" : " ----------------------------------\n") << endl;
  }
}

/////////////////////////////////////////////////////////////////////////////////////

void PatternOptimizer::updateStatCloseResults(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) {
  unsigned int iRefHit = omtfCand->getRefHitNumber();
  //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<omtfCandGp->key()<<" omtfCand->getGpResult()\n"<<omtfCand->getGpResult()<<std::endl;
  //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<exptCandGp->key()<<" exptResult\n"<<exptResult<<std::endl;

  if (omtfCandGp->key().theCharge != exptCandGp->key().theCharge) {
    /*cout<<__FUNCTION__<<":"<<__LINE__<<" omtfCandGp->key().theCharge != exptCandGp->key().theCharge"<<endl;
    cout<<__FUNCTION__<<":"<<__LINE__<<" "<<omtfCandGp->key()<<" omtfCand->getGpResult()\n"<<omtfCand->getGpResult()<<std::endl;
    cout<<__FUNCTION__<<":"<<__LINE__<<" "<<exptCandGp->key()<<" exptResult\n"<<exptResult<<std::endl;
    cout<<__FUNCTION__<<":"<<__LINE__<<" --------------------------------"<<std::endl;*/
    return;
  }

  int omtfCandPtCode = patternPtCodes[omtfCandGp->key().theNumber];
  int exptCandPtCode = patternPtCodes[exptCandGp->key().theNumber];

  GoldenPatternWithStat* secondBestGp = nullptr;
  for (auto& itGP : goldenPatterns) {
    if (itGP->key().thePt == 0 || omtfCandGp->key().theCharge != itGP->key().theCharge)
      continue;
    auto& result = itGP->getResults()[candProcIndx][iRefHit];
    if (result.getFiredLayerCnt() < omtfCand->getGpResult().getFiredLayerCnt()) {
      //itGP->gpProbabilityStat.Fill(0);
    } else {
      //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<itGP->key()<<" iRefHit "<<iRefHit<<" gpProbability "<<gpProbability<<" FiredLayerCnt "<<result.getFiredLayerCnt()<<std::endl;
      //itGP->gpProbabilityStat.Fill(result.getPdfSum());

      //finding second highest results, for which PdfSum differs from the omtfCandGp by 1 ,
      if (itGP.get() != omtfCandGp) {
        double delta = 1;
        if (itGP->key().theNumber > omtfCandGp->key().theNumber)
          delta = 0;

        int thisGpPtCode = patternPtCodes[itGP->key().theNumber];
        if (omtfCand->getGpResult().getPdfSum() - result.getPdfSum() == delta) {
          if (secondBestGp != nullptr) {
            if (abs(patternPtCodes[secondBestGp->key().theNumber] - exptCandPtCode) >
                abs(thisGpPtCode - exptCandPtCode)) {
              /*cout<<__FUNCTION__<<":"<<__LINE__<<" changing secondBestGp from "
                  <<secondBestGp->key()<<"\n"<<secondBestGp->getResults()[candProcIndx][iRefHit]<<"\nto "
                  <<itGP->key()<<result<<"\n omtfCand->getGpResult() "
                  <<omtfCandGp->key()<<"\n"<<omtfCand->getGpResult()<<"\n"
                  <<"exptCandGp"<<exptCandGp->key()<<endl;*/
              secondBestGp = itGP.get();
              // selecting secondBestGp that is closer to the exptCandGp
            }
          } else {
            secondBestGp = itGP.get();
          }

          //updating stat for any gp if the omtfCand is closer to the exptCand then this gp
          if (abs(omtfCandPtCode - exptCandPtCode) < abs(thisGpPtCode - exptCandPtCode)) {
            for (unsigned int iLayer = 0; iLayer < omtfCand->getGpResult().getStubResults().size(); iLayer++) {
              itGP->updateStat(
                  iLayer, result.getRefLayer(), result.getStubResults()[iLayer].getPdfBin(), goodSmaller, 1);
            }
          }
        }
      }
    }
  }

  if (secondBestGp) {
    auto& secondBestResult = secondBestGp->getResults()[candProcIndx][iRefHit];
    int secondBestGpPtCode = patternPtCodes[secondBestGp->key().theNumber];

    if (abs(omtfCandPtCode - exptCandPtCode) < abs(secondBestGpPtCode - exptCandPtCode)) {
      for (unsigned int iLayer = 0; iLayer < omtfCand->getGpResult().getStubResults().size(); iLayer++) {
        omtfCandGp->updateStat(iLayer,
                               omtfCand->getGpResult().getRefLayer(),
                               omtfCand->getGpResult().getStubResults()[iLayer].getPdfBin(),
                               goodBigger,
                               1);
        //secondBestGp->updateStat(iLayer, secondBestResult.getRefLayer(), secondBestResult.getHitPdfBins()[iLayer], goodSmaller, 1);
      }
    } else if ((omtfCandPtCode - exptCandPtCode) > 2 &&
               abs(omtfCandPtCode - exptCandPtCode) > abs(secondBestGpPtCode - exptCandPtCode)) {
      for (unsigned int iLayer = 0; iLayer < omtfCand->getGpResult().getStubResults().size(); iLayer++) {
        //cout<<__FUNCTION__<<":"<<__LINE__<<" badBigger "<<iLayer<<" "<<omtfCand->getGpResult().getRefLayer()<<" "<<omtfCand->getGpResult().getHitPdfBins()[iLayer]<<endl;
        //cout<<__FUNCTION__<<":"<<__LINE__<<" badSmaller "<<iLayer<<" "<<secondBestResult.getRefLayer()<<" "<<secondBestResult.getHitPdfBins()[iLayer]<<endl;
        omtfCandGp->updateStat(iLayer,
                               omtfCand->getGpResult().getRefLayer(),
                               omtfCand->getGpResult().getStubResults()[iLayer].getPdfBin(),
                               badBigger,
                               1);
        secondBestGp->updateStat(iLayer,
                                 secondBestResult.getRefLayer(),
                                 secondBestResult.getStubResults()[iLayer].getPdfBin(),
                                 badSmaller,
                                 1);
      }
      /*cout<<__FUNCTION__<<":"<<__LINE__<<" omtf cand id bad. secondBestGp "
          <<secondBestGp->key()<<"\n"<<secondBestGp->getResults()[candProcIndx][iRefHit]
          <<"\n omtfCand->getGpResult() "<<omtfCandGp->key()<<"\n"<<omtfCand->getGpResult()<<"\n"
          <<"exptCandGp"<<exptCandGp->key()<<endl<<"---------------------------------------------"<<endl;*/
    } else {  //equall
      //todo
    }
  }
}

void PatternOptimizer::updatePdfCloseResults() {
  cout << __FUNCTION__ << ":" << __LINE__ << endl;
  for (auto& gp : goldenPatterns) {
    if (gp->key().thePt == 0)
      continue;

    for (unsigned int iRefLayer = 0; iRefLayer < gp->getPdf()[0].size(); ++iRefLayer) {
      unsigned int selectedBiggerLayer = 0;
      unsigned int selectedBiggerBin = 0;
      int maxDiffBigger = 0;

      unsigned int selectedSmallerLayer = 0;
      unsigned int selectedSmallerBin = 0;
      int maxDiffSmaller = 0;

      //unsigned int refLayerLogicNumber = omtfConfig->getRefToLogicNumber()[iRefLayer];
      for (unsigned int iLayer = 0; iLayer < gp->getPdf().size(); ++iLayer) {
        /* if(refLayerLogicNumber == iLayer)
          continue; //not taking into account the class probability*/

        for (unsigned int iPdf = 1; iPdf < gp->getPdf()[iLayer][iRefLayer].size(); iPdf++) {
          int diffBigger =
              gp->statistics[iLayer][iRefLayer][iPdf][badBigger] - gp->statistics[iLayer][iRefLayer][iPdf][goodBigger];
          if (diffBigger > maxDiffBigger) {
            selectedBiggerLayer = iLayer;
            selectedBiggerBin = iPdf;
            maxDiffBigger = diffBigger;
          }

          int diffSmaller = gp->statistics[iLayer][iRefLayer][iPdf][badSmaller] -
                            gp->statistics[iLayer][iRefLayer][iPdf][goodSmaller];
          if (diffSmaller > maxDiffSmaller) {
            selectedSmallerLayer = iLayer;
            selectedSmallerBin = iPdf;
            maxDiffSmaller = diffSmaller;
          }
        }
      }

      if (selectedBiggerBin != 0 && maxDiffBigger > 5) {
        gp->pdfAllRef[selectedBiggerLayer][iRefLayer][selectedBiggerBin]--;
        cout << __FUNCTION__ << ":" << __LINE__ << " decreasing the pdf " << gp->key() << " iRefLayer " << iRefLayer
             << " layer " << selectedBiggerLayer << " bin " << selectedBiggerBin << " badBigger "
             << gp->statistics[selectedBiggerLayer][iRefLayer][selectedBiggerBin][badBigger] << " goodBigger "
             << gp->statistics[selectedBiggerLayer][iRefLayer][selectedBiggerBin][goodBigger] << " maxDiffBigger "
             << maxDiffBigger << endl;
      }

      if (selectedSmallerBin != 0 && maxDiffSmaller > 5) {
        gp->pdfAllRef[selectedSmallerLayer][iRefLayer][selectedSmallerBin]++;
        cout << __FUNCTION__ << ":" << __LINE__ << " increasing the pdf " << gp->key() << " iRefLayer " << iRefLayer
             << " layer " << selectedSmallerLayer << " bin " << selectedSmallerBin << " badSmaller "
             << gp->statistics[selectedSmallerLayer][iRefLayer][selectedSmallerBin][badSmaller] << " goodSmaller "
             << gp->statistics[selectedSmallerLayer][iRefLayer][selectedSmallerBin][goodSmaller] << " maxDiffSmaller "
             << maxDiffSmaller << endl;
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////

void PatternOptimizer::calculateThresholds(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) {
  //unsigned int iRefHit = omtfCand->getRefHitNumber();
  /*  cout<<__FUNCTION__<<":"<<__LINE__<<" "<<omtfCandGp->key()<<" omtfCand->getGpResult()\n"<<omtfCand->getGpResult()<<std::endl;
  cout<<__FUNCTION__<<":"<<__LINE__<<" "<<exptCandGp->key()<<" exptResult\n"<<exptResult<<std::endl;
  cout<<"all golden patterns:"<<endl;
  for(auto& gp : goldenPatterns) {
    if(omtfCand->getGpResult().getFiredLayerCnt() == gp->getResults()[candProcIndx][iRefHit].getFiredLayerCnt() )
    {
      cout<<gp->key()<<std::endl<<gp->getResults()[candProcIndx][iRefHit]<<std::endl;
    }
  }
  cout<<"----------------"<<endl;*/
  double gpProbability = exptResult.getGpProbability2();  //TODO chose  getGpProbability1 or getGpProbability2

  /*  exptCandGp->gpProbabilityStat[exptResult.getRefLayer()]->Fill(gpProbability);
  //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<exptCandGp->key()<<" iRefHit "<<iRefHit<<" gpProbability "<<gpProbability<<std::endl;
  return;*/

  ///////////////other version
  ///////////////////////////
  ///////////////////////////
  if (selectedPatNum != exptPatNum) {
    return;
  }

  if (omtfCandGp ==
      exptCandGp) {  // for the exptCandGp the threshold should be 0 in this iteration, so the ghosBuster should choose it, if the gp with the higher pt was not chose
    //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<exptCandGp->key()<<" iRefHit "<<iRefHit<<" gpProbability "<<gpProbability<<std::endl;
    exptCandGp->gpProbabilityStat[exptResult.getRefLayer()]->Fill(gpProbability);

    //debug
    /*if( gpProbability <= 0.002 ) {
      std::cout<<"omtfCand->getGpResult() "<<omtfCandGp->key()<<std::endl;
      std::cout<<std::endl<<omtfCand->getGpResult()<<std::endl;
      //int refHitNum = omtfCand->getRefHitNumber();
      std::cout<<"other gps results"<<endl;
      for(auto& gp : goldenPatterns) {
        if(omtfCand->getGpResult().getFiredLayerCnt() == gp->getResults()[candProcIndx][iRefHit].getFiredLayerCnt() )
        {
          cout<<gp->key()<<std::endl<<gp->getResults()[candProcIndx][iRefHit]<<std::endl;
        }
      }
      std::cout<<std::endl;
    }*/

  } else if (omtfCandGp->key().thePt >=
             exptCandGp->key().thePt) {  //= matters for the highest patterns with oposite signs
    exptCandGp->gpProbabilityStat[exptResult.getRefLayer()]->Fill(1.1);  //filling overflow bin in this case
    /*cout<<"--------OVERflow--------"<<endl;
    cout<<__FUNCTION__<<":"<<__LINE__<<" "<<omtfCandGp->key()<<" omtfCand->getGpResult()\n"<<omtfCand->getGpResult()<<std::endl;
    cout<<__FUNCTION__<<":"<<__LINE__<<" "<<exptCandGp->key()<<" exptResult\n"<<exptResult<<std::endl;*/
  } else {  //it can happened if the exptCandGp has lower FiredLayerCnt then the omtfCandG, or omtfCandGp (o has lower pt than exptCandGp because it is of opposite sign
    exptCandGp->gpProbabilityStat[exptResult.getRefLayer()]->Fill(-0.1);  //filling underflow bin in this case
    /*cout<<"--------underflow--------"<<endl;
    cout<<__FUNCTION__<<":"<<__LINE__<<" "<<omtfCandGp->key()<<" omtfCand->getGpResult()\n"<<omtfCand->getGpResult()<<std::endl;
    cout<<__FUNCTION__<<":"<<__LINE__<<" "<<exptCandGp->key()<<" exptResult\n"<<exptResult<<std::endl;*/
  }
}

/*
void PatternOptimizer::calculateThresholds(double targetEff) {
//  cout<<__FUNCTION__<<":"<<__LINE__<<" targetEff "<<targetEff<<std::endl;
//  TFile outfile("optimisedPats_2.root", "READ"); //FIXME the file name
//
//  ostringstream ostrName;
//  ostringstream ostrTitle;
//  for(unsigned int iPat = 0; iPat < this->myOmtfConfig->nGoldenPatterns(); iPat++) {
//    ostrName.str("");
//    ostrTitle.str("");
//    OMTFConfiguration::PatternPt patternPt = this->myOmtfConfig->getPatternPtRange(iPat);
//    ostrName<<"gpProbabilityStat_GP_"<<key().theNumber<<"_ptBinNum_"<<iPat;//<<"_muPtFrom_"<<patternPt.ptFrom<<"_GeV";
//    ostrTitle<<"gpProbabilityStat_GP_"<<key().theNumber<<"_ptBinNum_"<<iPat<<"_muPtFrom_"<<patternPt.ptFrom<<"_GeV";
//    if(patternPt.ptFrom > 0)
//      gpProbabilityStat.emplace_back(TH1I(ostrName.str().c_str(), ostrTitle.str().c_str(), 100, 0., 1.)); //TODO find proper range
//    else
//      gpProbabilityStat.emplace_back(TH1I(ostrName.str().c_str(), ostrTitle.str().c_str(), 1, 0., 1.)); //to save some memory, for empty patterns just "empty" hits
//  }


  //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<omtfCandGp->key()<<" omtfCand->getGpResult()\n"<<omtfCand->getGpResult()<<std::endl;
  //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<exptCandGp->key()<<" exptResult\n"<<exptResult<<std::endl;

  int takenMuCnt = 0;
  for(int iBin = goldenPatterns[currnetPtBatchPatNum]->gpProbabilityStat.GetNbinsX() + 1; iBin >= 0; iBin--) {//we include overflow bin, because it contains the muons found by the gps with higher pt
    takenMuCnt += goldenPatterns[currnetPtBatchPatNum]->gpProbabilityStat.GetBinContent(iBin);
    if(takenMuCnt >= targetEff * (double)(simMuFoundByOmtfPt->GetBinContent(currnetPtBatchPatNum + 1)) ) {
      float threshold = goldenPatterns[currnetPtBatchPatNum]->gpProbabilityStat.GetXaxis()->GetBinLowEdge(iBin);
      goldenPatterns[currnetPtBatchPatNum]->setThreshold(0, threshold);
      cout<<__FUNCTION__<<":"<<__LINE__<<" currnetPtBatchPatNum "<<currnetPtBatchPatNum<<" takenMuCnt "<<takenMuCnt
          <<" simMuFoundByOmtfPt "<<simMuFoundByOmtfPt->GetBinContent(currnetPtBatchPatNum + 1)
          <<" "<<goldenPatterns[currnetPtBatchPatNum]->gpProbabilityStat.Integral(0, 100000)
          <<" found threshold "<<threshold<<std::endl;
      break;
    }
  }

}
*/

void PatternOptimizer::calculateThresholds(double targetEff) {
  cout << __FUNCTION__ << ":" << __LINE__ << " targetEff " << targetEff << std::endl;
  //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<omtfCandGp->key()<<" omtfCand->getGpResult()\n"<<omtfCand->getGpResult()<<std::endl;
  //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<exptCandGp->key()<<" exptResult\n"<<exptResult<<std::endl;

  for (auto& gp : goldenPatterns) {
    if (gp->key().thePt == 0)
      continue;

    if (selectedPatNum != gp->key().theNumber)
      continue;

    for (unsigned int iRefLayer = 0; iRefLayer < gp->getPdf()[0].size(); ++iRefLayer) {
      int takenMuCnt = 0;
      int allMuCnt = gp->gpProbabilityStat[iRefLayer]->Integral(0, 100000);  //should include under and overflow

      for (int iBin = gp->gpProbabilityStat[iRefLayer]->GetNbinsX() + 1; iBin >= 0;
           iBin--) {  //we include overflow bin, because it contains the muons found by the gps with higher pt
        takenMuCnt += gp->gpProbabilityStat[iRefLayer]->GetBinContent(iBin);
        if (takenMuCnt >= targetEff * (double)(allMuCnt)) {
          float threshold = gp->gpProbabilityStat[iRefLayer]->GetXaxis()->GetBinLowEdge(iBin);
          gp->setThreshold(iRefLayer, threshold);
          cout << __FUNCTION__ << ":" << __LINE__ << gp->key() << " iRefLayer " << iRefLayer << " takenMuCnt "
               << takenMuCnt << " allMuCnt " << allMuCnt << " found threshold " << threshold << std::endl;
          break;
        }
      }
    }
  }
}

/*void PatternOptimizer::tuneClassProb(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) {
  unsigned int iRefHit = omtfCand->getRefHitNumber();
  //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<omtfCandGp->key()<<" omtfCand->getGpResult()\n"<<omtfCand->getGpResult()<<std::endl;
  //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<exptCandGp->key()<<" exptResult\n"<<exptResult<<std::endl;

  *
   * for the event with the muon pt > ptCut
   * for each low pt GP (pt < ptCut) histogramming the difference between the PdfSum() of this GP and the PdfSum of the omtfCandGp
   * if the

  //unsigned int iPdf = omtfConfig->nPdfBins()/2;

  int maxDiffBin = goldenPatterns[goldenPatterns.size()-1]->statisitics[0][0].size();

  //we have to find the gp above the threshold with the highest result
  double maxPdfSum = -1;
  auto& maxResult = goldenPatterns[selectedPatNum]->getResults()[candProcIndx][iRefHit];
  //unsigned int maxResultPat = 0;
  for(unsigned int iPat = selectedPatNum; iPat < goldenPatterns.size(); iPat++) {
    if(goldenPatterns[iPat]->key().thePt == 0)
      continue;

    if(goldenPatterns[iPat]->key().theCharge != goldenPatterns[exptPatNum]->key().theCharge) //exptPatNum should be = selectedPatNum
      continue;

    auto& result = goldenPatterns[iPat]->getResults()[candProcIndx][iRefHit];
    if(result.getFiredLayerCnt() == omtfCand->getGpResult().getFiredLayerCnt() ) {
      if(result.getPdfSum() > maxPdfSum) {
        maxPdfSum = result.getPdfSum();
        maxResult = result;
        //maxResultPat = iPat;
      }
    }
  }

  //if(omtfCandGp->key().thePt >= ptCut) //if commented, taking also the event in which the OMTF cand was below the thresh
  {
    for(unsigned int iPat = 0; iPat < goldenPatterns.size(); iPat++) {
      if(goldenPatterns[iPat]->key().thePt == 0)
        continue;

      if(goldenPatterns[iPat]->key().theCharge != goldenPatterns[exptPatNum]->key().theCharge)
        continue;

      unsigned int iRefLayer = omtfCand->getRefLayer();
      unsigned int refLayerLogicNumber = omtfConfig->getRefToLogicNumber()[iRefLayer];

      if(goldenPatterns[iPat]->key().thePt < ptCut) {
        auto& result = goldenPatterns[iPat]->getResults()[candProcIndx][iRefHit];

        double diff = maxDiffBin -1; //in the bin maxDiffBin -1 will go the events that are below the threshold

        if(maxPdfSum > 0  &&  result.getFiredLayerCnt() == omtfCand->getGpResult().getFiredLayerCnt()) {
          diff = maxResult.getPdfSum() - result.getPdfSum();

          if(result.getFiredLayerCnt() == omtfCand->getGpResult().getFiredLayerCnt() &&  diff < 0 ) {//debug
            cout<<__FUNCTION__<<":"<<__LINE__<<" diff = "<<diff<<std::endl;
            cout<<__FUNCTION__<<":"<<__LINE__<<" gp "<<goldenPatterns[iPat]->key()<<"\n"<<result<<std::endl;
            cout<<__FUNCTION__<<":"<<__LINE__<<" maxResultPat "<<goldenPatterns[maxResultPat]->key()<<"\n"<<maxResult<<std::endl;
          }

          diff = diff + maxDiffBin/2; //to allow for negative values

          if(diff >= (maxDiffBin-1) ) {
            diff = maxDiffBin -2; //overflows go here
          }
          else if (diff < 0) //watch out - this is already after adding the offset, so it is just to protect against seg fault!
            diff = 0;

          //debug
          if(diff < maxDiffBin/2) {
          cout<<__FUNCTION__<<":"<<__LINE__<<" diff < 0 !!!!!! diff = "<<diff<<std::endl;
          cout<<__FUNCTION__<<":"<<__LINE__<<" gp "<<goldenPatterns[iPat]->key()<<"\n"<<result<<std::endl;
          cout<<__FUNCTION__<<":"<<__LINE__<<" omtfCandGp "<<omtfCandGp->key()<<"\n"<<omtfCand->getGpResult()<<std::endl;
        }
        }
        //cout<<__FUNCTION__<<":"<<__LINE__<<goldenPatterns[iPat]->key()<<" iRefLayer "<<iRefLayer<<" diff = "<<diff<<std::endl;
        goldenPatterns[iPat]->statisitics[refLayerLogicNumber][iRefLayer][diff][ whatExptVal]++; //we use the  iPdf bin to put the diff histogram
      }
    }
  }
}*/

/*void PatternOptimizer::tuneClassProb(double targetEffLost) {
  cout<<__FUNCTION__<<":"<<__LINE__<<" targetEff "<<targetEffLost<<std::endl;
  cout<<__FUNCTION__<<": "<<__LINE__<<" Calculating P(C_k) "<<std::endl;
  unsigned int iPdf = omtfConfig->nPdfBins()/2;// <<(omtfConfig->nPdfAddrBits()-1);
  int maxDiffBin = goldenPatterns[goldenPatterns.size()-1]->statisitics[0][0].size();
  
  for(auto& gp : goldenPatterns) {
    if(gp->key().thePt == 0)
      continue;

    for(unsigned int iRefLayer = 0; iRefLayer < gp->getPdf()[0].size(); ++iRefLayer) {
      if(iRefLayer == 0 || iRefLayer == 2) //DT
        targetEffLost = 0.001;
      else if(iRefLayer == 5) //DT
        targetEffLost = 0.001;
      else if(iRefLayer == 1 || iRefLayer == 3 || iRefLayer == 4) //CSC & RE2/3
        targetEffLost = 0.035;
      else if(iRefLayer == 6 || iRefLayer == 7) //bRPC
        targetEffLost = 0.02;

      if(gp->key().thePt < ptCut && gp->key().theCharge == goldenPatterns[selectedPatNum]->key().theCharge) {
        cout<<endl;
        cout<<__FUNCTION__<<":"<<__LINE__<<gp->key()<<" RefLayer "<<iRefLayer<<std::endl;

        double cnt = 0;
        double norm = 0;
        unsigned int refLayerLogicNumber = omtfConfig->getRefToLogicNumber()[iRefLayer];

        for(int iDiff = 0; iDiff < maxDiffBin; iDiff++) {
          cout<<setw(4)<<(iDiff - maxDiffBin/2) <<" ";
        }
        cout<<endl;
        for(int iDiff = 0; iDiff < maxDiffBin; iDiff++) {
          norm += gp->statisitics[refLayerLogicNumber][iRefLayer][iDiff][whatExptVal];
          cout<<setw(4)<<gp->statisitics[refLayerLogicNumber][iRefLayer][iDiff][whatExptVal]<<" ";
        }
        cout<<endl;

        int iDiff = 0;
        for(; iDiff < maxDiffBin -1; iDiff++) { //last but one bin is overflow, last are the events with less fired planes
          cnt += gp->statisitics[refLayerLogicNumber][iRefLayer][iDiff][ whatExptVal];
          if( (cnt / norm) >= targetEffLost) {
            int diff = iDiff - maxDiffBin/2.;
            diff = diff - 1; //if thepdfSum is equal the lower pt GP is chosen, see <class GoldenPatternType> AlgoMuon OMTFSorter<GoldenPatternType>::sortRefHitResults
            cout<<__FUNCTION__<<":"<<__LINE__<<" cnt "<<cnt<<" norm "<<norm<<" current pdfVal "<<gp->pdfAllRef[refLayerLogicNumber][iRefLayer][iPdf]<<" diff "<<diff<<std::endl;

            if(gp->pdfAllRef[refLayerLogicNumber][iRefLayer][iPdf+1] == 0 || diff < 0) {//we are using the iPdf+1 to mark that the value was already updated
              gp->pdfAllRef[refLayerLogicNumber][iRefLayer][iPdf] += diff;

              cout<<__FUNCTION__<<":"<<__LINE__<<" new pdfVal "<<gp->pdfAllRef[refLayerLogicNumber][iRefLayer][iPdf]<<std::endl;
              if(gp->pdfAllRef[refLayerLogicNumber][iRefLayer][iPdf] <= 0) {
                cout<<__FUNCTION__<<":"<<__LINE__<<" pdf <= 0 !!!!!!!!!!!!!!!!!!!!!!"<<endl;;
              }

              gp->pdfAllRef[refLayerLogicNumber][iRefLayer][iPdf+1] = 1;
            }
            break;
          }
        }
        if(cnt == 0 && norm > 0)
          diff = maxDiffBin -1;
        if(iDiff == maxDiffBin -1) {
          cout<<__FUNCTION__<<":"<<__LINE__<<" diff not found, pdf not updated!!!! cnt "<<cnt<<" norm "<<norm<<" current pdf val "<<gp->pdfAllRef[refLayerLogicNumber][iRefLayer][iPdf]<<" iDiff "<<iDiff<<std::endl;
          //cout<<__FUNCTION__<<":"<<__LINE__<<" assuming diff is 52 !!!!!!!!!!!!!!!!!!!"<<endl;
          //gp->pdfAllRef[refLayerLogicNumber][iRefLayer][iPdf] += 52; ///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        }
      }

    }
  }

  for(unsigned int iRefLayer = 0; iRefLayer < goldenPatterns[0]->getPdf()[0].size(); ++iRefLayer) {
    for(auto& gp : goldenPatterns) {
      if(gp->key().thePt == 0)
        continue;
      if(gp->key().thePt < ptCut) {

        cout<<__FUNCTION__<<":"<<__LINE__<<gp->key()<<" RefLayer "<<iRefLayer<<std::endl;
      }
    }
  }
}*/

void PatternOptimizer::tuneClassProb(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) {
  //unsigned int iRefHit = omtfCand->getRefHitNumber();
  //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<omtfCandGp->key()<<" omtfCand->getGpResult()\n"<<omtfCand->getGpResult()<<std::endl;
  //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<exptCandGp->key()<<" exptResult\n"<<exptResult<<std::endl;

  unsigned int iRefLayer = omtfCand->getRefLayer();
  unsigned int refLayerLogicNumber = omtfConfig->getRefToLogicNumber()[iRefLayer];
  unsigned int iPdf = omtfConfig->nPdfBins() / 2;  //

  double ptSim = simMuon->momentum().pt();
  OMTFConfiguration::PatternPt expPatternPt = omtfConfig->getPatternPtRange(exptCandGp->key().theNumber);
  //cout<<__FUNCTION__<<":"<<__LINE__<<" ptSim "<<ptSim<<", expPatternPt: from "<<expPatternPt.ptFrom<<" to "<<expPatternPt.ptTo<<endl;

  if (expPatternPt.ptFrom < 5 || expPatternPt.ptFrom > 50)
    return;

  if ((expPatternPt.ptTo - 0.5) <= ptSim) {
    exptCandGp->statistics[refLayerLogicNumber][iRefLayer][iPdf][whatExptNorm]++;
    //cout<<__FUNCTION__<<":"<<__LINE__<<" filling lower bound norm ";
    if (omtfCandGp->key().thePt >= exptCandGp->key().thePt) {
      exptCandGp->statistics[refLayerLogicNumber][iRefLayer][iPdf][whatExptVal]++;
      //cout<<" end val";
    }
    //cout<<"\n\n";
  } else if ((expPatternPt.ptFrom + 0.5) > ptSim) {  //in this case the events goes to the prevous pattern
    unsigned int prevPatNum = omtfConfig->getPatternNum(expPatternPt.ptFrom - 0.25, exptCandGp->key().theCharge);
    auto& prevPat = goldenPatterns[prevPatNum];
    //cout<<__FUNCTION__<<":"<<__LINE__<<" filling upper bound for pattern "<<prevPat->key()<<" norm ";
    prevPat->statistics[refLayerLogicNumber][iRefLayer][iPdf + 1][whatExptNorm]++;
    if (omtfCandGp->key().thePt >= prevPat->key().thePt) {
      prevPat->statistics[refLayerLogicNumber][iRefLayer][iPdf + 1][whatExptVal]++;
      //cout<<" end val";
    }
    //cout<<"\n\n";
  }
}

void PatternOptimizer::tuneClassProb(double targetEff) {
  cout << __FUNCTION__ << ":" << __LINE__ << " targetEff " << targetEff << std::endl;
  cout << __FUNCTION__ << ": " << __LINE__ << " Calculating P(C_k) " << std::endl;
  unsigned int iPdf = omtfConfig->nPdfBins() / 2;  // <<(omtfConfig->nPdfAddrBits()-1);

  for (auto& gp : goldenPatterns) {
    if (gp->key().thePt == 0)
      continue;
    for (unsigned int iRefLayer = 0; iRefLayer < gp->getPdf()[0].size(); ++iRefLayer) {
      unsigned int refLayerLogicNumber = omtfConfig->getRefToLogicNumber()[iRefLayer];

      if (gp->statistics[refLayerLogicNumber][iRefLayer][iPdf][whatExptNorm] < 100 ||
          gp->statistics[refLayerLogicNumber][iRefLayer][iPdf + 1][whatExptNorm] < 100) {
        cout << __FUNCTION__ << ":" << __LINE__ << " " << gp->key() << " iRefLayer " << iRefLayer
             << " less than 100 in norm " << endl;
        continue;
      }

      double effLower = gp->statistics[refLayerLogicNumber][iRefLayer][iPdf][whatExptVal] /
                        gp->statistics[refLayerLogicNumber][iRefLayer][iPdf][whatExptNorm];

      double effUpper = gp->statistics[refLayerLogicNumber][iRefLayer][iPdf + 1][whatExptVal] /
                        gp->statistics[refLayerLogicNumber][iRefLayer][iPdf + 1][whatExptNorm];

      double effMean = (effLower + effUpper) / 2.;

      double delta = gp->pdfAllRef[refLayerLogicNumber][iRefLayer][iPdf] * ((targetEff - effMean));

      cout << __FUNCTION__ << ":" << __LINE__ << " " << gp->key() << " iRefLayer " << iRefLayer << " lowerCnt "
           << gp->statistics[refLayerLogicNumber][iRefLayer][iPdf][whatExptVal] << " lowerNorm "
           << gp->statistics[refLayerLogicNumber][iRefLayer][iPdf][whatExptNorm] << " effLower " << effLower
           << " upperCnt " << gp->statistics[refLayerLogicNumber][iRefLayer][iPdf + 1][whatExptVal] << " upperNorm "
           << gp->statistics[refLayerLogicNumber][iRefLayer][iPdf + 1][whatExptNorm] << " effUpper " << effUpper
           << " effMean " << effMean << " curPdfVal " << gp->pdfAllRef[refLayerLogicNumber][iRefLayer][iPdf]
           << " delta " << delta << endl;
      gp->pdfAllRef[refLayerLogicNumber][iRefLayer][iPdf] += delta;
    }
  }
}

void PatternOptimizer::updateStat(GoldenPatternWithStat* omtfCandGp,
                                  GoldenPatternWithStat* exptCandGp,
                                  double delta,
                                  double norm) {
  //cout<<__FUNCTION__<<":"<<__LINE__<<" exptCandGp Pt "<<exptCandGp->key().thePt<<std::endl;
  //cout<<__FUNCTION__<<":"<<__LINE__<<" omtfCandGp Pt "<<omtfCandGp->key().thePt<<" delta "<<delta<<" norm "<<norm<<std::endl;
  for (unsigned int iLayer = 0; iLayer < omtfCand->getGpResult().getStubResults().size(); iLayer++) {
    //updating statistic for the gp which should have fired
    int iBinExpt = exptResult.getStubResults()[iLayer].getPdfBin();
    if (iBinExpt != 0) {
      exptCandGp->updateStat(iLayer, exptResult.getRefLayer(), iBinExpt, whatExptVal, delta);
      exptCandGp->updateStat(iLayer, exptResult.getRefLayer(), iBinExpt, whatExptNorm, norm);
      //cout<<__FUNCTION__<<":"<<__LINE__<<" updating statistic for exptCandGp: iLayer "<<iLayer<<" RefLayer "<<exptResult.getRefLayer()<<" iBinExpt "<<iBinExpt<<" fired "<<exptResult.isLayerFired(iLayer)<<std::endl;
    }

    if (omtfCand->getGpResult().isLayerFired(iLayer) != 0) {
      //updating statistic for the gp which found the candidate
      //cout<<__FUNCTION__<<":"<<__LINE__<<" updating statistic for omtf gp "<<omtfCandGp<<std::endl;
      omtfCandGp->updateStat(iLayer,
                             omtfCand->getRefLayer(),
                             omtfCand->getGpResult().getStubResults()[iLayer].getPdfBin(),
                             whatOmtfVal,
                             -delta);
      omtfCandGp->updateStat(iLayer,
                             omtfCand->getRefLayer(),
                             omtfCand->getGpResult().getStubResults()[iLayer].getPdfBin(),
                             whatOmtfNorm,
                             norm);
      //cout<<__FUNCTION__<<":"<<__LINE__<<" updating statistic for omtfCandGp: iLayer "<<iLayer<<" RefLayer "<<omtfCand->getRefLayer()<<" iBinOmtf "<<omtfCand->getGpResult().getHitPdfBins()[iLayer]<<" fired "<<omtfCand->getGpResult().isLayerFired(iLayer)<<std::endl;
    }
  }
}

void PatternOptimizer::updateStatVoter_1(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) {
  double delta = 0;
  double norm = 1.;
  if (exptCandGp->key().thePt >= 51 && omtfCandGp->key().thePt < 51) {
    delta = 1.;  //1./10.;
  } else if (exptCandGp->key().thePt <= 33 && omtfCandGp->key().thePt >= 51) {
    delta = 6. / sqrt(exptCandGp->key().thePt);
    //norm = delta;
  }
  //std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
  /*if(exptCandGp->key().theCharge != omtfCandGp->key().theCharge) {
    delta = 2 * delta; //TODO what to do in this case????????
  }*/
  //delta = abs(delta);
  //delta *=delta;

  if (delta != 0) {
    updateStat(omtfCandGp, exptCandGp, delta, norm);
  }
}

void PatternOptimizer::updateStatPtLogDiff_1(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) {
  double delta = 0;
  if (exptCandGp->key().thePt > omtfCandGp->key().thePt) {
    delta = (log(exptCandGp->key().thePt) - log(omtfCandGp->key().thePt)) / log(omtfCandGp->key().thePt);
  } else {  //exptCandGp->key().thePt <= omtfCandGp->key().thePt
            /*if(exptCandGp->key().thePt > 50)
      delta = 0;
    else*/
    delta = (log(exptCandGp->key().thePt) - log(omtfCandGp->key().thePt)) / log(exptCandGp->key().thePt);
  }
  //std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
  if (exptCandGp->key().theCharge != omtfCandGp->key().theCharge) {
    delta = 2 * delta;  //TODO what to do in this case????????
  }
  //delta = abs(delta);
  delta *= delta;

  double norm = 1.;
  updateStat(omtfCandGp, exptCandGp, delta, norm);
}

void PatternOptimizer::updateStatPtDiff2_1(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) {
  double delta = 0;
  if (exptCandGp->key().thePt > omtfCandGp->key().thePt) {
    delta = (exptCandGp->key().thePt - omtfCandGp->key().thePt);
    delta /= 10.;
  } else {                                                        //exptCandGp->key().thePt <= omtfCandGp->key().thePt
    delta = (omtfCandGp->key().thePt - exptCandGp->key().thePt);  // / 2.; //watch out - the thePt is unsigned!!!!!!!
  }

  //std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
  if (omtfCandGp->key().thePt < 100 && exptCandGp->key().theCharge != omtfCandGp->key().theCharge) {
    delta = 2 * delta;  //TODO what to do in this case????????
  }
  //double norm = 1./((double)exptCandGp->key().thePt * (double)exptCandGp->key().thePt);
  //double norm = exp(-1.0 * (double)exptCandGp->key().thePt);
  double norm = 1.;
  //delta = abs(delta);
  delta *= delta;
  //delta *= norm;

  if (omtfCandGp->key().thePt != exptCandGp->key().thePt)
    updateStat(omtfCandGp, exptCandGp, delta, norm);
}

void PatternOptimizer::updateStatPtLogDiff_2(GoldenPatternWithStat* omtfCandGp, GoldenPatternWithStat* exptCandGp) {
  double delta = 0;
  if (exptCandGp->key().thePt > omtfCandGp->key().thePt) {
    delta = (log(exptCandGp->key().thePt - omtfCandGp->key().thePt)) / 1.5;
  } else {  //exptCandGp->key().thePt <= omtfCandGp->key().thePt
    if (exptCandGp->key().thePt > 50)
      delta = 0;
    else
      delta = (log(omtfCandGp->key().thePt) - log(exptCandGp->key().thePt));
  }
  //std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
  if (exptCandGp->key().theCharge != omtfCandGp->key().theCharge) {
    delta = 2 * delta;  //TODO what to do in this case????????
  }
  //delta = abs(delta);
  //delta *=delta;
  double norm = 1. / (double)exptCandGp->key().thePt;
  delta *= norm;
  updateStat(omtfCandGp, exptCandGp, delta, norm);
}

void PatternOptimizer::updatePdfsMean_1(GoldenPatternWithStat* gp,
                                        unsigned int& iLayer,
                                        unsigned int& iRefLayer,
                                        double& learingRate) {
  for (unsigned int iPdf = 1; iPdf < gp->getPdf()[iLayer][iRefLayer].size(); iPdf++) {
    double d = 0;
    if (gp->statistics[iLayer][iRefLayer][iPdf][whatExptNorm] != 0) {
      d += gp->statistics[iLayer][iRefLayer][iPdf][whatExptVal] /
           (double)gp->statistics[iLayer][iRefLayer][iPdf][whatExptNorm];
    }

    if (gp->statistics[iLayer][iRefLayer][iPdf][whatOmtfNorm] != 0)
      d += gp->statistics[iLayer][iRefLayer][iPdf][whatOmtfVal] /
           (double)gp->statistics[iLayer][iRefLayer][iPdf][whatOmtfNorm];

    d = d * learingRate;

    if (d > 0 && gp->key().thePt > 50 && gp->pdfAllRef[iLayer][iRefLayer][iPdf] == 0) {
      d = 0;  //don't add additional non zero beans for the high pt
    }

    gp->pdfAllRef[iLayer][iRefLayer][iPdf] += d;

    if (gp->pdfAllRef[iLayer][iRefLayer][iPdf] < 0)
      gp->pdfAllRef[iLayer][iRefLayer][iPdf] = 0;
    else if (gp->pdfAllRef[iLayer][iRefLayer][iPdf] > omtfConfig->pdfMaxValue()) {
      gp->pdfAllRef[iLayer][iRefLayer][iPdf] = omtfConfig->pdfMaxValue();
    }

    if (d != 0) {
      std::cout << __FUNCTION__ << ":" << __LINE__ << " " << gp->key() << " iLayer " << iLayer << " iRefLayer "
                << iRefLayer << " iBin " << iPdf << " pdfVal " << gp->pdfAllRef[iLayer][iRefLayer][iPdf] << " ExptVal "
                << gp->statistics[iLayer][iRefLayer][iPdf][whatExptVal] << " ExptNorm "
                << gp->statistics[iLayer][iRefLayer][iPdf][whatExptNorm] << " OmtfVal "
                << gp->statistics[iLayer][iRefLayer][iPdf][whatOmtfVal] << " OmtfNorm "
                << gp->statistics[iLayer][iRefLayer][iPdf][whatOmtfNorm] << " d " << d << std::endl;
    }
  }
}

void PatternOptimizer::updatePdfsMean_2(GoldenPatternWithStat* gp,
                                        unsigned int& iLayer,
                                        unsigned int& iRefLayer,
                                        double& learingRate) {
  for (unsigned int iPdf = 1; iPdf < gp->getPdf()[iLayer][iRefLayer].size(); iPdf++) {
    double d = 0;
    double norm = (double)gp->statistics[iLayer][iRefLayer][iPdf][whatExptNorm] +
                  (double)gp->statistics[iLayer][iRefLayer][iPdf][whatOmtfNorm];
    if (norm != 0)
      d += (gp->statistics[iLayer][iRefLayer][iPdf][whatExptVal] +
            gp->statistics[iLayer][iRefLayer][iPdf][whatOmtfVal]) /
           norm;

    d = d * learingRate;

    if (d > 0 && gp->key().thePt > 50 && gp->pdfAllRef[iLayer][iRefLayer][iPdf] == 0) {
      d = 0;  //don't add additional non zero beans for the high pt
    }

    gp->pdfAllRef[iLayer][iRefLayer][iPdf] += d;
    if (gp->pdfAllRef[iLayer][iRefLayer][iPdf] < 0)
      gp->pdfAllRef[iLayer][iRefLayer][iPdf] = 0;
    else if (gp->pdfAllRef[iLayer][iRefLayer][iPdf] > omtfConfig->pdfMaxValue()) {
      gp->pdfAllRef[iLayer][iRefLayer][iPdf] = omtfConfig->pdfMaxValue();
    }
    if (d != 0) {
      std::cout << __FUNCTION__ << ":" << __LINE__ << " " << gp->key() << " iLayer " << iLayer << " iRefLayer "
                << iRefLayer << " iBin " << iPdf << " pdfVal " << gp->pdfAllRef[iLayer][iRefLayer][iPdf] << " ExptVal "
                << gp->statistics[iLayer][iRefLayer][iPdf][whatExptVal] << " ExptNorm "
                << gp->statistics[iLayer][iRefLayer][iPdf][whatExptNorm] << " OmtfVal "
                << gp->statistics[iLayer][iRefLayer][iPdf][whatOmtfVal] << " OmtfNorm "
                << gp->statistics[iLayer][iRefLayer][iPdf][whatOmtfNorm] << " d " << d << std::endl;
    }
  }
}

void PatternOptimizer::updatePdfsVoter_1(GoldenPatternWithStat* gp,
                                         unsigned int& iLayer,
                                         unsigned int& iRefLayer,
                                         double& learingRate) {
  for (unsigned int iPdf = 1; iPdf < gp->getPdf()[iLayer][iRefLayer].size(); iPdf++) {
    double d = 0;
    double norm = (double)gp->statistics[iLayer][iRefLayer][iPdf][whatExptNorm] +
                  (double)gp->statistics[iLayer][iRefLayer][iPdf][whatOmtfNorm];
    if (norm != 0)
      d += (gp->statistics[iLayer][iRefLayer][iPdf][whatExptVal] +
            gp->statistics[iLayer][iRefLayer][iPdf][whatOmtfVal]);  // /norm;

    //d = d * learingRate;

    if (d > 0 && gp->key().thePt > 50 && gp->pdfAllRef[iLayer][iRefLayer][iPdf] == 0) {
      d = 0;  //don't add additional non zero beans for the high pt
    }

    if (gp->statistics[iLayer][iRefLayer][iPdf][whatExptNorm] > 5 ||
        gp->statistics[iLayer][iRefLayer][iPdf][whatOmtfNorm] > 5) {
      if (d >= 2)
        d = 1.0;
      else if (d <= 2)
        d = -1.0;
    } else
      d = 0;

    gp->pdfAllRef[iLayer][iRefLayer][iPdf] += d;
    if (gp->pdfAllRef[iLayer][iRefLayer][iPdf] < 0)
      gp->pdfAllRef[iLayer][iRefLayer][iPdf] = 0;
    else if (gp->pdfAllRef[iLayer][iRefLayer][iPdf] > omtfConfig->pdfMaxValue()) {
      gp->pdfAllRef[iLayer][iRefLayer][iPdf] = omtfConfig->pdfMaxValue();
    }
    if (d != 0) {
      std::cout << __FUNCTION__ << ":" << __LINE__ << " " << gp->key() << " iLayer " << iLayer << " iRefLayer "
                << iRefLayer << " iBin " << iPdf << " pdfVal " << gp->pdfAllRef[iLayer][iRefLayer][iPdf] << " ExptVal "
                << gp->statistics[iLayer][iRefLayer][iPdf][whatExptVal] << " ExptNorm "
                << gp->statistics[iLayer][iRefLayer][iPdf][whatExptNorm] << " OmtfVal "
                << gp->statistics[iLayer][iRefLayer][iPdf][whatOmtfVal] << " OmtfNorm "
                << gp->statistics[iLayer][iRefLayer][iPdf][whatOmtfNorm] << " d " << d << std::endl;
    }
  }
}

void PatternOptimizer::updatePdfsAnaDeriv(GoldenPatternWithStat* gp,
                                          unsigned int& iLayer,
                                          unsigned int& iRefLayer,
                                          double& learingRate) {
  for (unsigned int iPdf = 1; iPdf < gp->getPdf()[iLayer][iRefLayer].size(); iPdf++) {
    if (gp->pdfAllRef[iLayer][iRefLayer][iPdf] == 0)
      continue;

    double d = 0;
    d -= gp->statistics[iLayer][iRefLayer][iPdf][whatExptVal];
    d = d * learingRate;  //* gp->pdfAllRef[iLayer][iRefLayer][iPdf];
    if (nEventsRefL.at(iRefLayer))
      d /= 2.0 * nEventsRefL.at(iRefLayer);  // nEvents;
    else
      d = 0;

    if (abs(d) > abs(gp->pdfAllRef[iLayer][iRefLayer][iPdf]))
      d = (d > 0 ? gp->pdfAllRef[iLayer][iRefLayer][iPdf] : -1 * gp->pdfAllRef[iLayer][iRefLayer][iPdf]);

    if (d / gp->pdfAllRef[iLayer][iRefLayer][iPdf] > maxdPdf && gp->pdfAllRef[iLayer][iRefLayer][iPdf] > 0)
      maxdPdf = d / gp->pdfAllRef[iLayer][iRefLayer][iPdf];

    double pdfOld = gp->pdfAllRef[iLayer][iRefLayer][iPdf];
    gp->pdfAllRef[iLayer][iRefLayer][iPdf] += d;
    if (gp->pdfAllRef[iLayer][iRefLayer][iPdf] < 0) {
      gp->pdfAllRef[iLayer][iRefLayer][iPdf] = 1E-4;
      std::cout
          << "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ZEROWANIE SIE PFD  ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
          << std::endl;
    } else if (gp->pdfAllRef[iLayer][iRefLayer][iPdf] > omtfConfig->pdfMaxValue()) {
      gp->pdfAllRef[iLayer][iRefLayer][iPdf] = omtfConfig->pdfMaxValue();
    }
    if (d != 0) {
      std::cout << __FUNCTION__ << ":" << __LINE__ << " " << gp->key() << " iLayer " << setw(2) << iLayer
                << " iRefLayer " << iRefLayer << " iBin " << setw(3) << iPdf << " pdfVal " << setw(12) << left
                << gp->pdfAllRef[iLayer][iRefLayer][iPdf] << iPdf << " OldPdf " << setw(12) << left << pdfOld
                << " AnaDeriv " << setw(12) << left
                << gp->statistics[iLayer][iRefLayer][iPdf][whatExptVal]
                //<<" NumDeriv "<<gp->statisitics[iLayer][iRefLayer][iPdf][ whatExptNorm]
                << " AnaDeriv/2N " << setw(12) << left
                << gp->statistics[iLayer][iRefLayer][iPdf][whatExptVal] / 2.0 / nEventsRefL.at(iRefLayer) << " d "
                << setw(12) << left << d << " d/PdfVal " << setw(12) << left << ((pdfOld > 0) ? d / pdfOld : 0)
                << " nEvents " << setw(5) << nEventsRefL.at(iRefLayer) << " maxdPfd " << setw(12) << left << maxdPdf
                << std::endl;
    }
  }
}

void PatternOptimizer::updatePdfsNumDeriv(GoldenPatternWithStat* gp,
                                          unsigned int& iLayer,
                                          unsigned int& iRefLayer,
                                          double& learingRate) {
  for (unsigned int iPdf = 1; iPdf < gp->getPdf()[iLayer][iRefLayer].size(); iPdf++) {
    double d = 0;
    d -= gp->statistics[iLayer][iRefLayer][iPdf][whatExptNorm];
    d = d * learingRate;
    d /= 2.0 * nEvents;

    gp->pdfAllRef[iLayer][iRefLayer][iPdf] += d;
    if (gp->pdfAllRef[iLayer][iRefLayer][iPdf] < 0)
      gp->pdfAllRef[iLayer][iRefLayer][iPdf] = 0;
    else if (gp->pdfAllRef[iLayer][iRefLayer][iPdf] > omtfConfig->pdfMaxValue()) {
      gp->pdfAllRef[iLayer][iRefLayer][iPdf] = omtfConfig->pdfMaxValue();
    }
    if (d != 0) {
      std::cout << __FUNCTION__ << ":" << __LINE__ << " " << gp->key() << " iLayer " << iLayer << " iRefLayer "
                << iRefLayer << " iBin " << iPdf << " pdfVal " << gp->pdfAllRef[iLayer][iRefLayer][iPdf] << " AnaDeriv "
                << gp->statistics[iLayer][iRefLayer][iPdf][whatExptVal] << " NumDeriv "
                << gp->statistics[iLayer][iRefLayer][iPdf][whatExptNorm] << " OmtfVal "
                << gp->statistics[iLayer][iRefLayer][iPdf][whatOmtfVal] << " OmtfNorm "
                << gp->statistics[iLayer][iRefLayer][iPdf][whatOmtfNorm] << " d " << d << std::endl;
    }
  }
}

void PatternOptimizer::saveHists(TFile& outfile) {
  ostringstream ostr;
  ostr << "hitsCorrelation_refL_" << selRefL << "_firstL_" << selL1 << "_secondL_" << selL2;
  outfile.mkdir(ostr.str().c_str())->cd();
  for (auto& hist : deltaPhi1_deltaPhi2_hits) {
    if (hist)
      hist->Write();
  }
  for (auto& hist : deltaPhi1_deltaPhi2_omtf) {
    if (hist)
      hist->Write();
  }
}

void PatternOptimizer::initRateWeights() {
  rateWeights.assign(goldenPatterns.size(), 0);
  patternPtCodes.assign(goldenPatterns.size(), 0);

  //fist make the events probabity flat, i.e. to account for the content of the root fiels wiith events
  Float_t bins[] = {0.,  0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.,  6.,  7.,  8.,   10.,  12.,  14.,
                    16., 18., 20., 25., 30., 35., 40., 45., 50., 60., 70., 80., 90., 100., 120., 140., 1000.};
  Int_t binNum = sizeof(bins) / sizeof(Float_t) - 1;  // or just = 9
  TH1F* eventWeights = new TH1F("eventWeights", "eventWeights", binNum, bins);
  for (int iBin = 0; iBin < binNum - 1; iBin++) {
    eventWeights->Fill(bins[iBin], bins[iBin + 1] - bins[iBin]);
  }
  eventWeights->Fill(140, 20 * 50);

  eventRateWeights.assign(1000, 0);
  for (unsigned int i = 0; i < eventRateWeights.size(); i++) {
    double pt = i * 0.5;
    eventRateWeights[i] = eventWeights->GetBinContent(eventWeights->FindBin(pt)) * vxIntegMuRate(pt, 0.5, 0.82, 1.24);
    /*    if(i%10)
      std::cout<<"eventRateWeights "<<pt<<" Gev: rate "<<eventRateWeights[i]<< std::endl;*/
  }

  int positivePtCode = 1;
  int negativePtCode = 1;
  for (unsigned int patNum = 0; patNum < goldenPatterns.size(); patNum++) {
    if (goldenPatterns[patNum]->key().thePt == 0)
      continue;

    double ptFrom = omtfConfig->getPatternPtRange(patNum).ptFrom;
    double ptTo = omtfConfig->getPatternPtRange(patNum).ptTo;

    rateWeights[patNum] = vxIntegMuRate(ptFrom, ptTo - ptFrom, 0.82, 1.24);

    if (goldenPatterns[patNum]->key().theCharge == 1) {
      patternPtCodes[patNum] = positivePtCode;
      positivePtCode++;
    } else if (goldenPatterns[patNum]->key().theCharge == -1) {
      patternPtCodes[patNum] = negativePtCode;
      negativePtCode++;
    }

    //std::cout<<goldenPatterns[patNum]->key()<<" ptCode "<<patternPtCodes[patNum]<<" rate "<<rateWeights[patNum]<< std::endl;
  }
}

double PatternOptimizer::getEventRateWeight(double pt) {
  unsigned int bin = pt * 2;
  if (bin >= eventRateWeights.size())
    bin = eventRateWeights.size() - 1;

  return eventRateWeights[bin];
}
void PatternOptimizer::modifyPatterns1(double step) {
  cout << __FUNCTION__ << ": " << __LINE__ << " Correcting P(C_k) " << std::endl;
  unsigned int iPdf = omtfConfig->nPdfBins() / 2;  // <<(omtfConfig->nPdfAddrBits()-1);
  double delta = 0;
  for (unsigned int iRefLayer = 0; iRefLayer < goldenPatterns[0]->getPdf()[0].size(); ++iRefLayer) {
    unsigned int refLayerLogicNumber = omtfConfig->getRefToLogicNumber()[iRefLayer];
    if (iRefLayer == 0 || iRefLayer == 2)  //DT
      step = 1.5;
    else if (iRefLayer == 5)  //DT
      step = 1.5;
    else if (iRefLayer == 1)  //CSC
      step = 0.33333333;
    else if (iRefLayer == 3)  //CSC
      step = 0.5;
    else if (iRefLayer == 5)  //RE2/3
      step = 0.5;
    else if (iRefLayer == 6 || iRefLayer == 7)  //bRPC
      step = 1.5;

    cout << __FUNCTION__ << ":" << __LINE__ << " RefLayer " << iRefLayer << " step " << step << std::endl;
    for (int sign = -1; sign <= 1; sign++) {
      delta = 0;
      for (auto& gp : boost::adaptors::reverse(goldenPatterns)) {
        if (gp->key().thePt == 0 || gp->key().theCharge != sign)
          continue;
        int newPdfVal = gp->getPdf()[refLayerLogicNumber][iRefLayer][iPdf] + delta;
        gp->setPdfValue(newPdfVal, refLayerLogicNumber, iRefLayer, iPdf);
        delta += step;
      }
    }
  }
}
