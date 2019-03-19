/*
 * MuTimingModule.cc
 *
 *  Created on: Mar 7, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#include "L1Trigger/L1TMuonBayes/interface/MuTimingModule.h"
#include <math.h>
#include <iomanip>
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

MuTimingModule::MuTimingModule(const ProcConfigurationBase* config): config(config),
  timigTo1_Beta(config->nLayers() )
{
  for(unsigned int iLayer = 0; iLayer < timigTo1_Beta.size(); ++iLayer) {
    //TODO rolls, eta and bins can depend on layer
    timigTo1_Beta[iLayer].assign(rolls, std::vector<std::vector<int> >(etaBins, std::vector<int>(timingBins)) );
  }
}

MuTimingModule::~MuTimingModule() {
  // TODO Auto-generated destructor stub
}

void MuTimingModule::process(AlgoMuonBase* algoMuon) {
  std::vector<unsigned int> one_betaHist(betaBins, 0);

  LogTrace("omtfEventPrintout")<<__FUNCTION__<<":"<<__LINE__<<std::endl;
  for(auto& stubResult : algoMuon->getStubResults() ) {
    if(!stubResult.getValid())
      continue;

    if(stubResult.getMuonStub()->type != MuonStub::RPC) //TODO add aother types if available
      continue;

    unsigned int layer = stubResult.getMuonStub()->logicLayer;
    unsigned int roll =  stubResult.getMuonStub()->roll;
    unsigned int etaBin = etaHwToEtaBin(algoMuon->getEtaHw(), stubResult.getMuonStub());
    unsigned int hitTimingBin = timingToTimingBin(stubResult.getMuonStub()->timing);

    unsigned int one_beta = timigTo1_Beta.at(layer).at(roll).at(etaBin).at(hitTimingBin);

    LogTrace("omtfEventPrintout")<<__FUNCTION__<<":"<<__LINE__<<" layer "<<layer<<" roll "<<roll<<" etaBin "<<etaBin<<" hitTiming "<<stubResult.getMuonStub()->timing<<" hitTimingBin "<<hitTimingBin<<" = one_beta "<<one_beta<<std::endl;

    if(one_beta != 0) {
      //pseudo-Gauss
      one_betaHist.at(one_beta) +=2;
      if(one_beta != 0)
        one_betaHist.at(one_beta-1) +=1;
      if(one_beta != one_betaHist.size() -1)
        one_betaHist.at(one_beta+1) +=1;

    }
  }

  //finding max one_betaHist bin
  unsigned int maxVal = 0;
  unsigned int maxBin = 0;
  unsigned int entriesSum = 0;
  for(unsigned int i = 0; i < one_betaHist.size(); i++) {
    if(one_betaHist[i] > maxVal) {
      maxVal = one_betaHist[i];
      maxBin = i;
    }
    entriesSum += one_betaHist[i];
  }

  float beta = one_betaBinToBeta(maxBin);
  LogTrace("omtfEventPrintout")<<__FUNCTION__<<":"<<__LINE__<<" maxBin "<<maxBin<<" maxVal "<<maxVal<<" beta "<<beta<<std::endl;
  if(maxVal >= 3) {
    algoMuon->setBeta(beta);
  }
  else
    algoMuon->setBeta(0);
}

unsigned int MuTimingModule::etaHwToEtaBin(int trackEtaHw, const MuonStubPtr& muonStub) const {
  int etaBin = 2 + trackEtaHw - (muonStub->etaHw - muonStub->etaSigmaHw);
  etaBin = etaBin / 8;
  if(etaBin < 0)
    return 0;
  if(etaBin >= (int)etaBins)
    return etaBins - 1;
  else
    return etaBin;
}

//betaBin = 0 reserved for no-beta, betaBin = 1 - beta = 1
unsigned int MuTimingModule::betaTo1_betaBin(double beta) const {
  if(beta == 0)
    return 0;

  unsigned int one_betaBin = round((1./beta - 1.) * 4. + 1.);
  if(one_betaBin >= betaBins)
    return betaBins-1;
  return one_betaBin;
}

float MuTimingModule::one_betaBinToBeta(unsigned int one_betaBin) const {
  if(one_betaBin == 0)
    return 0;
  float beta = 1./( (one_betaBin - 1)/4. + 1);
  return beta;
}


unsigned int MuTimingModule::timingToTimingBin(int timing) const {
  int bin = (timing)/2;
  if(bin < 0)
    return 0;
  if (bin >= (int)timingBins)
    return timingBins -1;
  return bin;
}
