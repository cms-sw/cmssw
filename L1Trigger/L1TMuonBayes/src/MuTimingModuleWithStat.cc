/*
 * MuTimingModuleWithStat.cc
 *
 *  Created on: Mar 8, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#include "L1Trigger/L1TMuonBayes/interface/MuTimingModuleWithStat.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TF1.h"

MuTimingModuleWithStat::MuTimingModuleWithStat(const ProcConfigurationBase* config): MuTimingModule(config),
timigVs1_BetaHists(config->nLayers()) {
  TFileDirectory subDir = fileService->mkdir("timingModule");

  //[layer][wheel_ring][etaBin][timing][1_Beta]
  for(unsigned int iLayer = 0; iLayer < timigTo1_Beta.size(); ++iLayer) {
    for(unsigned int iRoll = 0; iRoll < timigTo1_Beta[iLayer].size(); ++iRoll) {
      timigVs1_BetaHists[iLayer].emplace_back();
      for(unsigned int iEtaBin = 0; iEtaBin < timigTo1_Beta[iLayer][iRoll].size(); ++iEtaBin) {
        /*if(iLayer < 13 || iLayer > 22) {
          timigVs1_BetaHists[iLayer][iRoll].emplace_back(nullptr);
          continue;
        }*/
        std::ostringstream name;
        name<<"timingHist_layer_"<<iLayer<<"_roll_"<<iRoll<<"_eta_"<<iEtaBin;
        //LogTrace("l1tMuBayesEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" creating timigVs1_BetaHists "<<name.str()<<std::endl;
        TH2I* hist = subDir.make<TH2I>(name.str(). c_str(), name.str(). c_str(), 50, -10, 90, betaBins, -0.5, betaBins -0.5);
        hist->GetXaxis()->SetTitle("hit timing [ns]");
        hist->GetYaxis()->SetTitle("(1/beta-1) * 4 + 1");
        timigVs1_BetaHists[iLayer][iRoll].emplace_back(hist);
      }
    }
  }

  betaDist = subDir.make<TH1I>("betaDist", "betaDist", 100, 0, 1);
  betaDist->GetXaxis()->SetTitle("beta");
}

MuTimingModuleWithStat::~MuTimingModuleWithStat() {
}

void MuTimingModuleWithStat::process(AlgoMuonBase* algoMuon) {
  if(algoMuon->getSimBeta() == 0) {
    LogTrace("l1tMuBayesEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" algoMuon->getSimBeta() == 0 "<<std::endl;
    return;
  }

  betaDist->Fill(algoMuon->getSimBeta());

  unsigned int one_beta = betaTo1_betaBin(algoMuon->getSimBeta() );
  LogTrace("l1tMuBayesEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" algoMuon SimBeta "<<algoMuon->getSimBeta()<<" one_beta "<<one_beta<<std::endl;

  for(auto& stubResult : algoMuon->getStubResults() ) {
    if(!stubResult.getValid())
      continue;

    unsigned int layer = stubResult.getMuonStub()->logicLayer;
    unsigned int roll =  stubResult.getMuonStub()->roll;
    unsigned int etaBin = etaHwToEtaBin(algoMuon->getEtaHw(), stubResult.getMuonStub());

    int hitTiming = stubResult.getMuonStub()->timing;

    LogTrace("l1tMuBayesEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" layer "<<layer
        <<" algoMuon eta "<<algoMuon->getEtaHw()<<" MuonStub eta "<<stubResult.getMuonStub()->etaHw<<"  etaSigma "<<stubResult.getMuonStub()->etaSigmaHw
        <<" etaBin "<<etaBin<<" hitTiming "<<stubResult.getMuonStub()->timing<<std::endl;

    auto timigVs1_BetaHist = timigVs1_BetaHists.at(layer).at(roll).at(etaBin);
    if(timigVs1_BetaHist)
      timigVs1_BetaHist->Fill(hitTiming, one_beta);
  }
}


void MuTimingModuleWithStat::generateCoefficients() {
  for(unsigned int iLayer = 0; iLayer < timigTo1_Beta.size(); ++iLayer) {
    for(unsigned int iRoll = 0; iRoll < timigTo1_Beta[iLayer].size(); ++iRoll) {
      for(unsigned int iEtaBin = 0; iEtaBin < timigTo1_Beta[iLayer][iRoll].size(); ++iEtaBin) {
        auto timigVs1_BetaHist = timigVs1_BetaHists.at(iLayer).at(iRoll).at(iEtaBin);
        if(!timigVs1_BetaHist)
          continue;

        for(unsigned int iBetaBin = 0; iBetaBin < betaBins; iBetaBin++) {
          for(unsigned int iTimingBin = 0; iTimingBin < timingBins; iTimingBin++) {
            timigTo1_Beta.at(iLayer).at(iRoll).at(iEtaBin).at(iTimingBin).at(iBetaBin) = 0; //cleanig previous values
          }
        }

        //timigVs1_BetaHist->Sumw2();
        if(timigVs1_BetaHist->Integral() <= 0) {
          //LogTrace("l1tMuBayesEventPrint")<<__FUNCTION__<<": "<<__LINE__<<" iLayer "<<iLayer<<" iRoll "<<iRoll<<" iEtaBin "<<iEtaBin<<" - no entries, coefficients not calculated"<<std::endl;
          continue;
        }

        for(unsigned int iBetaBin = 0; iBetaBin < betaBins; iBetaBin++) {
          //Normalize pdf in each betaBin separately, to get p(timing | beta, eta)
          std::ostringstream ostr;
          ostr<<timigVs1_BetaHist->GetName()<<"_ptBin_"<<iBetaBin;
          TH1D* timingHistInBetaBin = timigVs1_BetaHist->ProjectionX(ostr.str().c_str(), iBetaBin +1, iBetaBin +1); //+1 Because the bins in root hist are counted from 1

          timingHistInBetaBin->SetTitle(ostr.str().c_str());

          //timingHistInBetaBin->Sumw2();
          if(timingHistInBetaBin->Integral() <= 0) {
            edm::LogVerbatim("l1tMuBayesEventPrint")<<__FUNCTION__<<": "<<__LINE__<<" iLayer "<<iLayer<<" iEtaBin "<<iEtaBin<<" iBetaBin "<<iBetaBin<<" - no entries, coefficients not calculated"<<std::endl;
            continue;
          }

          timingHistInBetaBin->Scale(1./timingHistInBetaBin->Integral());

          const double minPdfVal = 0.01;
          const double minPlog =  log(minPdfVal);
          const double pdfMaxLogVal = 3; //the maximum value tha the logPdf can have (n.b. logPdf = pdfMaxLogVal - log(pdfVal) * pdfMaxLogVal / minPlog)


          for(int iTimingHistBin = 1; iTimingHistBin <= timingHistInBetaBin->GetXaxis()->GetNbins(); iTimingHistBin++) {
            double pdfVal = timingHistInBetaBin->GetBinContent(iTimingHistBin);
            double logPdf = 0;
            //double error = 0;
            if(pdfVal >= minPdfVal) {
              logPdf = pdfMaxLogVal - log(pdfVal) * pdfMaxLogVal / minPlog;
            }

            int timing = timingHistInBetaBin->GetXaxis()->GetBinLowEdge(iTimingHistBin);
            unsigned int timingBin = timingToTimingBin(timing);

            timigTo1_Beta.at(iLayer).at(iRoll).at(iEtaBin).at(timingBin).at(iBetaBin) = + round(logPdf); //in the timingHistInBetaBin there are bins with negative timing. after round it will not be very good
            //pdfHistInPtBin->SetBinContent(iTimingHistBin, logPdf);

            if(timigTo1_Beta.at(iLayer).at(iRoll).at(iEtaBin).at(timingBin).at(iBetaBin) > pdfMaxLogVal)
              timigTo1_Beta.at(iLayer).at(iRoll).at(iEtaBin).at(timingBin).at(iBetaBin) = pdfMaxLogVal; //should be not needed if the above is corrected

            edm::LogVerbatim("l1tMuBayesEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" layer "<<iLayer<<" roll "<<iRoll<<" etaBin "<<iEtaBin<<" iBetaBin "<<iBetaBin
                <<" iTimingHistBin "<<iTimingHistBin<<" iTimingBin "<<timingBin<<" timing "<<timing<<" logPdf "<<logPdf<<std::endl;
          }

        }
      }
    }
  }
}


/*void MuTimingModuleWithStat::generateCoefficients() {
  for(unsigned int iLayer = 0; iLayer < timigTo1_Beta.size(); ++iLayer) {
    for(unsigned int iRoll = 0; iRoll < timigTo1_Beta[iLayer].size(); ++iRoll) {
      for(unsigned int iEtaBin = 0; iEtaBin < timigTo1_Beta[iLayer][iRoll].size(); ++iEtaBin) {
        auto timigVs1_BetaHist = timigVs1_BetaHists.at(iLayer).at(iRoll).at(iEtaBin);
        if(!timigVs1_BetaHist)
          continue;

        timigVs1_BetaHist->Sumw2();
        if(timigVs1_BetaHist->Integral() <= 0) {
          //LogTrace("l1tMuBayesEventPrint")<<__FUNCTION__<<": "<<__LINE__<<" iLayer "<<iLayer<<" iRoll "<<iRoll<<" iEtaBin "<<iEtaBin<<" - no entries, coefficients not calculated"<<std::endl;
          continue;
        }

        for(int iTimingBin = 1; iTimingBin <= timigVs1_BetaHist->GetXaxis()->GetNbins(); iTimingBin++) {
          //finding max bin for each iTimingBin
          int maxVal = 0;
          int one_beta = 0; //0 is not possibile value
          for(int iBetaBin = 1; iBetaBin <= timigVs1_BetaHist->GetYaxis()->GetNbins(); iBetaBin++) {
            double binVal = timigVs1_BetaHist->GetBinContent(iTimingBin, iBetaBin);
            if(binVal > maxVal) {
              maxVal = binVal;
              one_beta = timigVs1_BetaHist->GetYaxis()->GetBinCenter(iBetaBin);
            }
          }
          int timing =  timigVs1_BetaHist->GetXaxis()->GetBinLowEdge(iTimingBin);
          if((iLayer < 13 || iLayer > 22)  && timing < 6) {//for RPPC assuming everything with timing  6 is muon, due to timing resolution TODO optimize when real resolution available
            one_beta = 1;
            //if timing < 0 hitTimingBin then is 0
          }
          unsigned int hitTimingBin = timingToTimingBin(timing);
          timigTo1_Beta.at(iLayer).at(iRoll).at(iEtaBin).at(hitTimingBin) = one_beta;
          edm::LogVerbatim("l1tMuBayesEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" layer "<<iLayer<<" roll "<<iRoll<<" etaBin "<<iEtaBin<<" timing "<<timing
              <<" hitTimingBin "<<hitTimingBin<<" calculated one_beta "<<one_beta<<std::endl;

        }

      }
    }
  }
}*/
