#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternWithStat.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternBase.h"

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"
#include "TH1.h"
#include <string>

#include "boost/multi_array/base.hpp"
#include "boost/multi_array/subarray.hpp"

////////////////////////////////////////////////////
////////////////////////////////////////////////////
/*GoldenPatternWithStat::GoldenPatternWithStat(const Key& aKey, const OMTFConfiguration * omtfConfig): GoldenPattern(aKey, omtfConfig) {
  //GoldenPattern::reset();
  //reset();
}*/

GoldenPatternWithStat::GoldenPatternWithStat(const Key& aKey,
                                             unsigned int nLayers,
                                             unsigned int nRefLayers,
                                             unsigned int nPdfAddrBits)
    : GoldenPatternWithThresh(aKey, nLayers, nRefLayers, nPdfAddrBits),
      statistics(boost::extents[nLayers][nRefLayers][(1 << nPdfAddrBits) * 8]
                               [STAT_BINS])  //TODO remove *8!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //gpProbabilityStat( ("gpProbabilityStat_GP_" + to_string(key().theNumber)).c_str(), ("gpProbabilityStat_GP_" + to_string(key().theNumber)).c_str(), 1000, 0., 1.) //TODO find proper range
      {

      };

GoldenPatternWithStat::GoldenPatternWithStat(const Key& aKey, const OMTFConfiguration* omtfConfig)
    : GoldenPatternWithThresh(aKey, omtfConfig),
      statistics(boost::extents[omtfConfig->nLayers()][omtfConfig->nRefLayers()][omtfConfig->nPdfBins()][STAT_BINS]){

      };

void GoldenPatternWithStat::updateStat(
    unsigned int iLayer, unsigned int iRefLayer, unsigned int iBin, unsigned int what, double value) {
  statistics[iLayer][iRefLayer][iBin][what] += value;
  //std::cout<<__FUNCTION__<<":"<<__LINE__<<" iLayer "<<iLayer<<" iRefLayer "<<iRefLayer<<" iBin "<<iBin<<" what "<<what<<" value "<<value<<std::endl;
}
////////////////////////////////////////////////////
////////////////////////////////////////////////////
/*void GoldenPatternWithStat::updatePdfs(double learingRate) {
  //double f = 1;
  for(unsigned int iLayer = 0; iLayer < getPdf().size(); ++iLayer) {
    for(unsigned int iRefLayer=0; iRefLayer < getPdf()[iLayer].size(); ++iRefLayer) {
      for(unsigned int iPdf = 1; iPdf < getPdf()[iLayer][iRefLayer].size(); iPdf++) {
        double d = 0;
        if(statisitics[iLayer][iRefLayer][iPdf][whatSimNorm] != 0)
          d -= statisitics[iLayer][iRefLayer][iPdf][whatSimVal]/(double)statisitics[iLayer][iRefLayer][iPdf][whatSimNorm];

        if(statisitics[iLayer][iRefLayer][iPdf][whatOmtfNorm] != 0)
          d += statisitics[iLayer][iRefLayer][iPdf][whatOmtfVal]/(double)statisitics[iLayer][iRefLayer][iPdf][whatOmtfNorm] ;

        d = d * learingRate;
        pdfAllRef[iLayer][iRefLayer][iPdf] += d;
        if(d != 0) {
          std::cout<<__FUNCTION__<<":"<<__LINE__<<" "<< key()<<" iLayer "<<iLayer<<" iRefLayer "<<iRefLayer<<" iBin "<<iPdf<<" pdfVal "<<pdfAllRef[iLayer][iRefLayer][iPdf]<<" d "<<d<<std::endl;
        }
      }
    }
  }
}*/

std::ostream& operator<<(std::ostream& out, const GoldenPatternWithStat& aPattern) {
  /*  out <<"GoldenPatternWithStat "<< aPattern.theKey <<std::endl;
  out <<"Number of reference layers: "<<aPattern.meanDistPhi[0].size()
          <<", number of measurement layers: "<<aPattern.pdfAllRef.size()
          <<std::endl;

  if(!aPattern.meanDistPhi.size()) return out;
  if(!aPattern.pdfAllRef.size()) return out;

  out<<"Mean dist phi per layer:"<<std::endl;
  for (unsigned int iRefLayer=0;iRefLayer<aPattern.meanDistPhi[0].size();++iRefLayer){
    out<<"Ref layer: "<<iRefLayer<<" (";
    for (unsigned int iLayer=0;iLayer<aPattern.meanDistPhi.size();++iLayer){   
      out<<std::setw(3)<<aPattern.meanDistPhi[iLayer][iRefLayer]<<"\t";
    }
    out<<")"<<std::endl;
  }

  if(aPattern.meanDistPhiCounts.size()){
    out<<"Counts number per layer:"<<std::endl;
    for (unsigned int iRefLayer=0;iRefLayer<aPattern.meanDistPhi[0].size();++iRefLayer){
      out<<"Ref layer: "<<iRefLayer<<" (";
      for (unsigned int iLayer=0;iLayer<aPattern.meanDistPhi.size();++iLayer){   
        out<<aPattern.meanDistPhiCounts[iLayer][iRefLayer]<<"\t";
      }
      out<<")"<<std::endl;
    }
  }

  unsigned int nPdfAddrBits = 7;
  out<<"PDF per layer:"<<std::endl;
  for (unsigned int iRefLayer=0;iRefLayer<aPattern.pdfAllRef[0].size();++iRefLayer){
    out<<"Ref layer: "<<iRefLayer;
    for (unsigned int iLayer=0;iLayer<aPattern.pdfAllRef.size();++iLayer){   
      out<<", measurement layer: "<<iLayer<<std::endl;
      for (unsigned int iRefPhiB=0; iRefPhiB < aPattern.pdfAllRef[iLayer][iRefLayer].size(); ++iRefPhiB) {
        for(unsigned int iPdf=0;iPdf<exp2(nPdfAddrBits);++iPdf){
          out<<std::setw(2)<<aPattern.pdfAllRef[iLayer][iRefLayer][iRefPhiB][iPdf]<<" ";
        }
      }
      out<<std::endl;
    }
  }*/

  return out;
}

void GoldenPatternWithStat::initGpProbabilityStat() {
  unsigned int nRefLayers = pdfAllRef[0].size();

  unsigned int binNum = 1000 + 100 + 10 - 2;
  Double_t* bins = new Double_t[binNum + 1];

  Double_t lowerEdge = 0;
  for (unsigned int i = 0; i <= binNum; i++) {
    bins[i] = lowerEdge;
    if (lowerEdge < 0.001) {
      //std::cout<<__FUNCTION__<<":"<<__LINE__<<" i "<<i<<"lowerEdge "<<lowerEdge<<std::endl;
      lowerEdge += 0.00001;
    } else if (lowerEdge >= 0.999) {
      //std::cout<<__FUNCTION__<<":"<<__LINE__<<" i "<<i<<"lowerEdge "<<lowerEdge<<std::endl;
      lowerEdge += 0.0001;
    } else
      lowerEdge += 0.001;
  }

  for (unsigned int iRefLayer = 0; iRefLayer < nRefLayers; ++iRefLayer) {
    gpProbabilityStat.push_back(
        new TH1I(("gpProbabilityStat_GP_" + to_string(key().theNumber) + "_refLay_" + to_string(iRefLayer)).c_str(),
                 ("gpProbabilityStat GP " + to_string(key().theNumber) + " refLayer " + to_string(iRefLayer)).c_str(),
                 binNum,
                 bins));
  }

  delete bins;
}
