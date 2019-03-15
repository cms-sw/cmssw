/*
 * PdfModule.cc
 *
 *  Created on: Feb 4, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */


#include "L1Trigger/L1TMuonBayes/interface/MuCorrelator/PdfModule.h"

PdfModule::PdfModule(MuCorrelatorConfigPtr& config): IPdfModule(config), coefficients(config->nLayers() )  {
  init();
}

void PdfModule::init() {
  for(unsigned int iLayer = 0; iLayer < coefficients.size(); ++iLayer) {

    unsigned int etaBins = 1;
    if(config->isEndcapLayer(iLayer)) {
      etaBins = config->nEtaBins();
    }

    coefficients[iLayer].assign(etaBins, std::vector<std::vector<std::vector<int> > >(config->nRefLayers(), std::vector<std::vector<int> >(config->nPtBins(), std::vector<int>(3)) ));
    //[layer][etaBin][ptBin][coefficient index]
  }

  //some dummy values, TODO
  for(unsigned int iLayer = 0; iLayer < coefficients.size(); ++iLayer) {
    for(unsigned int iEtaBin = 0; iEtaBin < coefficients[iLayer].size(); ++iEtaBin) {
      for(unsigned int iRefLayer = 0; iRefLayer < coefficients[iLayer][iEtaBin].size(); ++iRefLayer) {
        for(unsigned int iPtBin = 0; iPtBin < coefficients[iLayer][iEtaBin][iRefLayer].size(); ++iPtBin) {
          coefficients[iLayer][iEtaBin][iRefLayer][iPtBin][0] = 400;
          coefficients[iLayer][iEtaBin][iRefLayer][iPtBin][1] = 0;
          coefficients[iLayer][iEtaBin][iRefLayer][iPtBin][2] = -1;
        }
      }
    }
  }
}

float PdfModule::getPdfVal(unsigned int layer, unsigned int etaBin, unsigned int refLayer, unsigned int ptBin, int pdfBin) {
  std::vector<int>& coeff = coefficients.at(layer).at(etaBin).at(refLayer).at(ptBin);
  return  (-1) * ( (coeff.at(2) * pdfBin * ((int64_t)pdfBin) )>>bitShift ) + coeff.at(0);
}

float PdfModule::getExtrapolation(unsigned int layer, unsigned int etaBin, unsigned int refLayer, unsigned int ptBin) {
  return coefficients.at(layer).at(etaBin).at(refLayer).at(ptBin).at(1);
}

void PdfModule::processStubs(const MuonStubsInput& muonStubs, unsigned int layer, const TrackingTriggerTrackPtr& ttTrack, const MuonStubPtr refStub, AlgoTTMuonPtr algoTTMuon) {
  unsigned int ptBin = ttTrack->getPtBin();
  int charge = -ttTrack->getCharge(); //we inverse the charge since the positive tracks bends into negative phi direction, and we wan to to have positive values of pdfBin when the coeff.at(1) is 0
  //== positive values of coeff.at(1)

/*  MuonStub::Type stubType = muonStubs.getMuonStubs()[layer][0]->type; //TODO
  if(stubType == MuonStub::DT_PHI_ETA || stubType == MuonStub::CSC_PHI_ETA || stubType == MuonStub::DT_HIT ) {
    throw runtime_error("PdfModule::processStub: handling the stubs of the type DT_PHI_ETA and CSC_PHI_ETA and DT_HIT not implemented");
  }*/

  unsigned int etaBin = 0;
  if(config->isEndcapLayer(layer))
    etaBin = ttTrack->getEtaBin();
  //assuming that in the barrel the muon track bending is not dependent on eta

  //std::cout<<__FUNCTION__<<":"<<__LINE__<<" layer "<<layer<<" ptBin "<<ptBin<<" etaBin "<<etaBin<<std::endl;

  unsigned int refLayerNum = 0;
  if(refStub) {
    refLayerNum = config->logLayerToRefLayar(refStub->logicLayer, etaBin);
  }

  int extrapolation = getExtrapolation(layer, etaBin, refLayerNum, ptBin);

  int minPdfBin = config->nPhiBins();
  MuonStubPtr selectedStub;

  for(auto& stub : muonStubs.getMuonStubs()[layer]) {
    int pdfBin = 0;
    if(config->isPhiLayer(layer)) {
      int refPhi = ttTrack->getPhiHw();
      if(refLayerNum != 0) { //refLayerNum = 0 means that no refLayer is used
        if(layer != refStub->logicLayer) //for the layer == refLayer the refPhi is the phi of the ttTrack, otherwise phi of the refHit
          refPhi = refStub->phiHw;
      }
      pdfBin = charge * config->foldPhi(stub->phiHw - refPhi ) - extrapolation; //extrapolation is meanDistPhi, TODO check if the formula is OK
    }
    else  if(config->isEtaLayer(layer))
      pdfBin = stub->etaHw - ttTrack->getEtaHw() - extrapolation; //do we need for eta to extrapolate, i.e. to subtract coeff.at(1)?

    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" processed stub\n"<<(*stub)<<" pdfBin "<<pdfBin<<std::endl;
    if( abs(pdfBin) <= abs(minPdfBin) ) {
      minPdfBin = pdfBin;
      selectedStub = stub;
    }
  }

  if(!selectedStub)
    return;

  //TODO optional correction for the different distance from vertex of the different DT superlayers, decide if needed. Might be also applied at the step of selecting the closest stub
  /*if(isEndcap == 0) {
      unsigned int quality = selectedStub->qualityHw;
      if(quality == 1)
        minPdfBin += coeff.at(3);
      else if(quality == 2)
        minPdfBin += coeff.at(4);
      //no change if quality is 3 //TODO check the quality definition
    }
    else {
      //endcap
    }*/

  float pdfVal = getPdfVal(layer, etaBin, refLayerNum, ptBin, minPdfBin);
  //std::cout<<__FUNCTION__<<":"<<__LINE__<<" selectedStub\n"<<(*selectedStub)<<" minPdfBin "<<minPdfBin<<" pdfVal "<<pdfVal<<std::endl;

  //if pdfVal <= 0, the stub result is not valid, todo - maybe then simply dont add it? on the other hand may be needed for debugging or optimization
  bool phiHitValid = pdfVal > 0;
  algoTTMuon->addStubResult(pdfVal, phiHitValid, minPdfBin, layer, selectedStub );

  //handling of the phiB
  if(selectedStub->type == MuonStub::DT_PHI || selectedStub->type == MuonStub::DT_PHI_ETA ) {
    //assuming that the corresponding bandig layer is just layer + 1, TODO maybe add function getBandingLayer(layer)
    int extrapolation = getExtrapolation(layer + 1, etaBin, refLayerNum, ptBin);
    int pdfBin = charge * selectedStub->phiBHw - extrapolation;
    pdfVal = getPdfVal(layer +1, etaBin, refLayerNum, ptBin, pdfBin);

    algoTTMuon->addStubResult(pdfVal, phiHitValid & (pdfVal > 0), pdfBin, layer + 1, selectedStub );
  }

  return;
}



