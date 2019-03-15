/*
 * AlgoTTMuon.cc
 *
 *  Created on: Feb 1, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#include "L1Trigger/L1TMuonBayes/interface/MuCorrelator/AlgoTTMuon.h"

#include <iomanip>


void AlgoTTMuon::addStubResult(float pdfVal, bool valid, int pdfBin, int layer, MuonStubPtr stub) {
  if(valid) {
    //stubResults.emplace_back(pdfVal, valid, pdfBin, layer, stub);
    pdfSum += pdfVal;
    firedLayerBits.set(layer);
  }
  stubResults[layer] = StubResult(pdfVal, valid, pdfBin, layer, stub);

  //stub result is added evevn thought it is not valid since this might be needed for debugging or optimization
}

std::ostream & operator << (std::ostream &out, const AlgoTTMuon& algoTTMuon) {
  out <<"algoTTMuon: "<<std::endl;
  out<<(*algoTTMuon.ttTrack)<<std::endl;
  out <<"firedLayerBits: "<<algoTTMuon.firedLayerBits<<" pdfSum "<<algoTTMuon.pdfSum<<" quality "<<algoTTMuon.quality<<std::endl;
  out <<"stubResults: "<<std::endl;
  for(auto& stubResult : algoTTMuon.stubResults) {
    if(stubResult.getMuonStub() ) {
      out <<"layer "<<std::setw(2)<<stubResult.getLayer()<<" valid "<<stubResult.getValid()<<" pdfBin "<<std::setw(5)<<stubResult.getPdfBin()
        <<" pdfVal "<<std::setw(8)<<stubResult.getPdfVal()<<" "<<(*stubResult.getMuonStub())<<std::endl;
    }
  }

  if(algoTTMuon.refStub)
    out <<"refStub: "<<algoTTMuon.refStub<<std::endl;

  out<<std::endl;
  return out;
}
