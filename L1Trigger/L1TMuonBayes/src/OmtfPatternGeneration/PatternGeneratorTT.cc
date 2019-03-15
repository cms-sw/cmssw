/*
 * PatternGeneratorTT.cc
 *
 *  Created on: Oct 17, 2018
 *      Author: kbunkow
 */

#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFProcessorTTMerger.h>
#include <L1Trigger/L1TMuonBayes/interface/OmtfPatternGeneration/PatternGeneratorTT.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TFile.h"

PatternGeneratorTT::PatternGeneratorTT(const edm::ParameterSet& edmCfg, const OMTFConfiguration* omtfConfig, std::vector<std::shared_ptr<GoldenPatternWithStat> >& gps):
  PatternOptimizerBase(edmCfg, omtfConfig, gps)
{
  //adding new patterns!!!!!!!!!!!!
  //GoldenPatternWithStat(const Key & aKey, const OMTFConfiguration* omtfConfig);

  edm::LogImportant("PatternGeneratorTT") << "PatternGeneratorTT: adding new patterns and modifying existing!!!!!" << std::endl;

  gps[46]->setKeyPt(45);
  gps[47]->setKeyPt(49);
  gps[52]->setKeyPt(53);

  gps[42]->setKeyPt(45);
  gps[43]->setKeyPt(49);
  gps[48]->setKeyPt(53);

  auto pos = gps.begin();
  gps.insert(pos, make_shared<GoldenPatternWithStat>(Key(0, 0, -1, 0), omtfConfig)); pos = gps.begin();
  gps.insert(pos, make_shared<GoldenPatternWithStat>(Key(0, 0, -1, 0), omtfConfig)); pos = gps.begin();
  gps.insert(pos, make_shared<GoldenPatternWithStat>(Key(0, 8, -1, 0), omtfConfig)); pos = gps.begin();
  gps.insert(pos, make_shared<GoldenPatternWithStat>(Key(0, 7, -1, 0), omtfConfig)); pos = gps.begin();

  gps.insert(pos, make_shared<GoldenPatternWithStat>(Key(0, 0, 1, 0), omtfConfig)); pos = gps.begin();
  gps.insert(pos, make_shared<GoldenPatternWithStat>(Key(0, 0, 1, 0), omtfConfig)); pos = gps.begin();
  gps.insert(pos, make_shared<GoldenPatternWithStat>(Key(0, 8, 1, 0), omtfConfig)); pos = gps.begin();
  gps.insert(pos, make_shared<GoldenPatternWithStat>(Key(0, 7, 1, 0), omtfConfig)); pos = gps.begin();

  goldenPatterns = gps;

  unsigned int i = 0;
  for(auto& gp : goldenPatterns) {
      gp->setKeyNumber(i++);

      if(gp->key().thePt == 0)
        continue;

      gp->reset();

      int statBinsCnt = gp->getPdf()[0][0].size() * 8 *2;
      gp->iniStatisitics(statBinsCnt, 1); //TODO
  }

  edm::LogImportant("PatternGeneratorTT") << "reseting golden pattern !!!!!" << std::endl;
  GoldenPatternResult::setFinalizeFunction(3);

  //TODO remove, it is needed only for the patterns generation
/*  for(auto& gp : goldenPatterns) {
    for(unsigned int iLayer = 0; iLayer < gp->getPdf().size(); ++iLayer) {
      for(unsigned int iRefLayer = 0; iRefLayer < gp->getPdf()[iLayer].size(); ++iRefLayer) {
        unsigned int refLayerLogicNum = omtfConfig->getRefToLogicNumber()[iRefLayer];
        if(refLayerLogicNum == iLayer) {
          for(unsigned int iBin = 0; iBin < gp->getPdf()[iLayer][iRefLayer].size(); iBin++) {
            gp->pdfAllRef[iLayer][iRefLayer][iBin] = 1;
          }
        }
      }
    }
  }*/

  ptDeltaPhiHists.resize(2);
  for(unsigned int iCharge = 0; iCharge <= 1; iCharge++) {
    for(unsigned int iLayer = 0; iLayer < omtfConfig->nLayers(); ++iLayer) { //for the moment filing only ref layer, remove whe
      if(iLayer == 0 || iLayer == 2 || iLayer == 4 || iLayer == 6 || iLayer == 7 || iLayer == 10 || iLayer == 11  || iLayer == 16 || //refLayars
         iLayer == 1 || iLayer == 3 || iLayer == 5  ) //banding layers
      {
        ostringstream name;
        name<<"ptDeltaPhiHist_ch_"<<iCharge<<"_Layer_"<<iLayer;
        int phiFrom = -10;
        int phiTo   = 1000;
        int phiBins = phiTo - phiFrom;

        if(iCharge == 1) {
          phiFrom = -1000;
          phiTo = 10;
        }

        TH2I* ptDeltaPhiHist = new TH2I(name.str().c_str(), name.str().c_str(), 400, 0, 200, phiBins, phiFrom -0.5, phiTo -0.5);
        //cout<<"BinLowEdge "<<ptDeltaPhiHist->GetYaxis()->GetBinLowEdge(100)<<" BinUpEdge "<<ptDeltaPhiHist->GetYaxis()->GetBinUpEdge(100);
        ptDeltaPhiHists[iCharge].push_back(ptDeltaPhiHist);
      }
      else
        ptDeltaPhiHists[iCharge].push_back(nullptr);
    }
  }
}

PatternGeneratorTT::~PatternGeneratorTT() {
  // TODO Auto-generated destructor stub
}

void PatternGeneratorTT::updateStat() {
  //cout<<__FUNCTION__<<":"<<__LINE__<<" omtfCand "<<*omtfCand<<std::endl;;
  TTAlgoMuon* ttAlgoMuon = dynamic_cast<TTAlgoMuon*>(omtfCand.get());
  if(!ttAlgoMuon) {
    cout<<__FUNCTION__<<":"<<__LINE__<<" ttAlgoMuon is null"<<std::endl;
    throw runtime_error("ttAlgoMuon is null");
  }
  //TODO go through all refLayers in the omtfCandGp

  GoldenPatternWithStat* omtfCandGp = static_cast<GoldenPatternWithStat*>(omtfCand->getGoldenPatern());

  unsigned int iCharge = omtfCand->getCharge();
  if(iCharge != 1)
    iCharge = 0;

  int pdfMiddle = 1<<(omtfConfig->nPdfAddrBits()-1);

  for(auto& gpResult : ttAlgoMuon->getGpResults()) {
    unsigned int iRefLayer =  gpResult->getRefLayer();
    unsigned int refLayerLogicNum = omtfConfig->getRefToLogicNumber()[iRefLayer];

    //cout<<gpResult;
    if(gpResult->getFiredLayerCnt() >= 3 )
    {
      //cout<<__FUNCTION__<<":"<<__LINE__<<" updating statistic"<<std::endl;
      for(unsigned int iLayer = 0;  iLayer < gpResult->getStubResults().size(); iLayer++) {
        //updating statistic for the gp which should have fired

        int phiDist = gpResult->getStubResults()[iLayer].getPdfBin();
        phiDist += omtfCandGp->meanDistPhiValue(iLayer, iRefLayer) - pdfMiddle; //removing the shift applied in the GoldenPatternBase::process1Layer1RefLayer

        if(ptDeltaPhiHists[iCharge][iLayer] != nullptr &&
            (iLayer == refLayerLogicNum || omtfConfig->getLogicToLogic().at(iLayer) == (int)refLayerLogicNum) )
          ptDeltaPhiHists[iCharge][iLayer]->Fill(ttAlgoMuon->getTtTrack().getPt(), phiDist);

        phiDist += omtfCandGp->getStatistics()[iLayer][iRefLayer].size()/2;

        //cout<<__FUNCTION__<<":"<<__LINE__<<" iLayer "<<iLayer<<" phiDist "<<phiDist<<std::endl;
        if( phiDist >= 0 && phiDist < (int)(omtfCandGp->getStatistics()[iLayer][iRefLayer].size()) ) {
          //updating statistic for the gp which found the candidate
          //cout<<__FUNCTION__<<":"<<__LINE__<<" updating statistic for omtf gp "<<omtfCandGp<<std::endl;
          omtfCandGp->updateStat(iLayer, iRefLayer, phiDist, 0, 1);
        }
      }
    }
  }
}

void PatternGeneratorTT::observeEventEnd(const edm::Event& iEvent) {
  if(simMuon == 0 || omtfCand->getGoldenPatern() == 0)//no sim muon or empty candidate
    return;

  PatternOptimizerBase::observeEventEnd(iEvent);

  updateStat();
}

void PatternGeneratorTT::endJob() {

  upadatePdfs();

  PatternOptimizerBase::endJob();
}

void PatternGeneratorTT::upadatePdfs() {
  cout<<__FUNCTION__<<": "<<__LINE__<<endl;

  int minHitCnt = 300;
  //Calculating meanDistPhi
  for(auto& gp : goldenPatterns) {
    if(gp->key().thePt == 0)
      continue;
    for(unsigned int iLayer = 0; iLayer < gp->getPdf().size(); ++iLayer) {

      for(unsigned int iRefLayer = 0; iRefLayer < gp->getPdf()[iLayer].size(); ++iRefLayer) {
        //unsigned int refLayerLogicNum = omtfConfig->getRefToLogicNumber()[iRefLayer];
        //if(refLayerLogicNum == iLayer)
        {

          //calculate meanDistPhi
          double meanDistPhi = 0;
          double count = 0;
          for(unsigned int iBin = 0; iBin < gp->getStatistics()[iLayer][iRefLayer].size(); iBin++) {
            meanDistPhi +=  iBin * gp->getStatistics()[iLayer][iRefLayer][iBin][0];
            count       +=         gp->getStatistics()[iLayer][iRefLayer][iBin][0];
          }
          if(count != 0) {
            meanDistPhi /= count;

            meanDistPhi -= (gp->getStatistics()[iLayer][iRefLayer].size() / 2);

            if(count < minHitCnt)
              meanDistPhi = 0;
            cout<<__FUNCTION__<<": "<<__LINE__<<" "<<gp->key()<<" iLayer "<<iLayer<<" iRefLayer "<<iRefLayer<<" count "<<count<<" meanDistPhi "<<meanDistPhi<<endl;
            gp->setMeanDistPhiValue(round(meanDistPhi), iLayer, iRefLayer);
          }
        }
      }
    }
  }

//averaging the meanDistPhi for the gp belonging to the same group
  OMTFConfiguration::vector2D mergedPartters = omtfConfig->getMergedPatterns(goldenPatterns);
  for(unsigned int iLayer = 0; iLayer < goldenPatterns.at(0)->getPdf().size(); ++iLayer) {
    for(unsigned int iRefLayer = 0; iRefLayer < goldenPatterns.at(0)->getPdf()[iLayer].size(); ++iRefLayer) {
      //unsigned int refLayerLogicNum = omtfConfig->getRefToLogicNumber()[iRefLayer];
      //if(refLayerLogicNum == iLayer)
      {
        for(unsigned int iGroup = 0; iGroup < mergedPartters.size(); iGroup++) {
          double meanDistPhi = 0;
          int mergedCnt = 0;
          for(unsigned int i = 0; i < mergedPartters[iGroup].size(); i++) {
            auto gp = goldenPatterns.at(mergedPartters[iGroup][i]).get();
            meanDistPhi += gp->meanDistPhiValue(iLayer, iRefLayer);
            if(gp->meanDistPhiValue(iLayer, iRefLayer) != 0)
              mergedCnt++;
          }
          meanDistPhi /= mergedCnt; //because for some gps the statistics can be too low, and then the meanDistPhiValue is 0, so it should not contribute
          for(unsigned int i = 0; i < mergedPartters[iGroup].size(); i++) {
            auto gp = goldenPatterns.at(mergedPartters[iGroup][i]).get();
            gp->setMeanDistPhiValue(round(meanDistPhi), iLayer, iRefLayer);
            cout<<__FUNCTION__<<": "<<__LINE__<<" "<<gp->key()<<" iLayer "<<iLayer<<" iRefLayer "<<iRefLayer<<" meanDistPhi after averaging "<<meanDistPhi<<endl;
          }
        }
      }
    }
  }


  //setting the DistPhiBitShift i.e. grouping of the pdfBins
  for(auto& gp : goldenPatterns) {
    if(gp->key().thePt == 0)
      continue;
    for(unsigned int iLayer = 0; iLayer < gp->getPdf().size(); ++iLayer) {
      for(unsigned int iRefLayer = 0; iRefLayer < gp->getPdf()[iLayer].size(); ++iRefLayer) {
        if(gp->getDistPhiBitShift(iLayer, iRefLayer) ) {
          throw runtime_error(string(__FUNCTION__) + ":" + to_string(__LINE__) + "gp->getDistPhiBitShift(iLayer, iRefLayer) != 0 -  cannot change DistPhiBitShift then!!!!");
        }

        unsigned int refLayerLogicNum = omtfConfig->getRefToLogicNumber()[iRefLayer];
        if(refLayerLogicNum == iLayer)
        {
          //TODO the thresholds depends on the pdfWidth i.e. address bits count
          if( omtfConfig->hwPtToGev(gp->key().thePt) < 5  ) {
            if(iLayer == 0 || iLayer == 10 || iLayer == 11)
              gp->setDistPhiBitShift(3, iLayer, iRefLayer);
            else
              gp->setDistPhiBitShift(2, iLayer, iRefLayer);
          }
          else if( omtfConfig->hwPtToGev(gp->key().thePt) < 10  ) {
            gp->setDistPhiBitShift(1, iLayer, iRefLayer);
          }
        }
        else if( ( (gp->key().thePt <= 8) && (iLayer == 1 || iLayer == 3 || (iLayer == 5) ) ) ) {
          gp->setDistPhiBitShift(2, iLayer, iRefLayer);
        }
        else if( ( (gp->key().thePt <= 10) && (                              (iLayer == 5) ) ) ) {
          gp->setDistPhiBitShift(2, iLayer, iRefLayer);
        }
        else if( ( (gp->key().thePt <= 10) && (iLayer == 1 || iLayer == 3                  ) ) ) {
          gp->setDistPhiBitShift(1, iLayer, iRefLayer);
        }
      }
    }
  }

  //calculating the pdfs
  for(auto& gp : goldenPatterns) {
    if(gp->key().thePt == 0)
      continue;
    for(unsigned int iLayer = 0; iLayer < gp->getPdf().size(); ++iLayer) {

      for(unsigned int iRefLayer = 0; iRefLayer < gp->getPdf()[iLayer].size(); ++iRefLayer) {
        //unsigned int refLayerLogicNum = omtfConfig->getRefToLogicNumber()[iRefLayer];
        //if(refLayerLogicNum == iLayer)
        {
          double norm = 0;
          for(unsigned int iBin = 0; iBin < gp->getStatistics()[iLayer][iRefLayer].size(); iBin++) {
            norm += gp->getStatistics()[iLayer][iRefLayer][iBin][0];
          }

          int pdfMiddle = gp->getPdf()[iLayer][iRefLayer].size() / 2;
          int statBinGroupSize = 1<<gp->getDistPhiBitShift(iLayer, iRefLayer);
          for(unsigned int iBinPdf = 0; iBinPdf < gp->getPdf()[iLayer][iRefLayer].size(); iBinPdf++) {
            double pdfVal = 0;
            int groupedBins = 0;
            for(int i = 0; i < statBinGroupSize; i++) {
              int iBinStat = statBinGroupSize * ((int)(iBinPdf) - pdfMiddle) + i + gp->meanDistPhiValue(iLayer, iRefLayer);

              iBinStat += (gp->getStatistics()[iLayer][iRefLayer].size()/2);

              if(iBinStat >= 0 && iBinStat < (int)gp->getStatistics()[iLayer][iRefLayer].size() ) {
                pdfVal += gp->getStatistics()[iLayer][iRefLayer][iBinStat][0];
                groupedBins++;
                //cout<<__FUNCTION__<<": "<<__LINE__<<" "<<gp->key()<<" iLayer "<<iLayer<<" iBinStat "<<iBinStat<<" iBinPdf "<<iBinPdf<<" statVal "<<gp->getStatistics()[iLayer][iRefLayer][iBinStat][0]<<endl;
              }
            }
            if(norm > minHitCnt) {
              pdfVal /= norm;
            }
            else
              pdfVal = 0;

            const double minPlog =  log(omtfConfig->minPdfVal());
            const double pdfMaxVal = omtfConfig->pdfMaxValue();

            int digitisedVal = 0;
            if(pdfVal >= omtfConfig->minPdfVal()) {
              digitisedVal = rint(pdfMaxVal - log(pdfVal) / minPlog * pdfMaxVal);
            }

            gp->setPdfValue(digitisedVal, iLayer, iRefLayer, iBinPdf );
            //cout<<__FUNCTION__<<": "<<__LINE__<<" "<<gp->key()<<" iLayer "<<iLayer<<" iBinPdf "<<iBinPdf<<" pdfVal "<<pdfVal<<" digitisedVal "<<digitisedVal<<endl;
          }

        }
      }
    }
  }
}

void PatternGeneratorTT::saveHists(TFile& outfile) {
  outfile.mkdir("ptDeltaPhiHists" )->cd();
  for(unsigned int iCharge = 0; iCharge <= 1; iCharge++) {
    for(unsigned int iLayer = 0; iLayer < omtfConfig->nLayers(); ++iLayer) { //for the moment filing only ref layer, remove whe
      if(ptDeltaPhiHists[iCharge][iLayer]) {
        ptDeltaPhiHists[iCharge][iLayer]->Write();
      }
    }
  }
}

