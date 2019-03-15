#include "L1Trigger/L1TMuonBayes/interface/Omtf/GoldenPatternResult.h"
#include "L1Trigger/L1TMuonBayes/interface/Omtf/OMTFConfiguration.h"
#include <iostream>
#include <ostream>
#include <iomanip>
#include <cmath>


////////////////////////////////////////////
////////////////////////////////////////////
int GoldenPatternResult::finalizeFunction = 0;

////////////////////////////////////////////
////////////////////////////////////////////
GoldenPatternResult::GoldenPatternResult(const OMTFConfiguration * omtfConfig):  valid(false), omtfConfig(omtfConfig) {
  if(omtfConfig)
    init(omtfConfig);
}

////////////////////////////////////////////
////////////////////////////////////////////
/*void GoldenPatternResult::configure(const OMTFConfiguration * omtfConfig) {
  myOmtfConfig = omtfConfig;
  assert(myOmtfConfig != 0);
  reset();
}*/
////////////////////////////////////////////
////////////////////////////////////////////

void GoldenPatternResult::set(int refLayer_, int phi, int eta, int refHitPhi) {
  if( isValid() && this->refLayer != refLayer_) {
    std::cout<<__FUNCTION__<<" "<<__LINE__<<" this->refLayer "<<this->refLayer<<" refLayer_ "<<refLayer_<<std::endl;
  }
  assert( !isValid() || this->refLayer == refLayer_);

  this->refLayer = refLayer_;
  this->phi = phi;
  this->eta = eta;
  this->refHitPhi = refHitPhi;
}

/*void GoldenPatternResult::setLayerResult(unsigned int iLayer, LayerResult layerResult) {
  pdfValues.at(iLayer) = layerResult.pdfVal;
  if(layerResult.valid) {
    firedLayerBits |= (1<< iLayer);stubResults
  }
  hitPdfBins[iLayer] = layerResult.pdfBin;
  hits[iLayer] = layerResult.hit;
  if(layerResult.valid || layerResult.pdfVal)
    std::cout<<__FUNCTION__<<" "<<__LINE__<<" iLayer "<<iLayer<<" refLayer "<<refLayer<<" pdfBin "<<layerResult.pdfBin<<" val "<<layerResult.pdfVal<<" valid "<<layerResult.valid<<std::endl;
  //pdfSum += pdfVal; - this cannot be done here, because the pdfVal for the banding layer must be added only
  //if hit in the corresponding phi layer was accpeted (i.e. its pdfVal > 0. therefore it is done in finalise()
}*/

void GoldenPatternResult::setStubResult(float pdfVal, bool valid, int pdfBin, int layer, MuonStubPtr stub) {
  if(valid) {
    //pdfSum += pdfVal;
    //firedLayerBits.set(layer);
    firedLayerBits |= (1<< layer);
  }
  stubResults[layer] = StubResult(pdfVal, valid, pdfBin, layer, stub);

  //stub result is added evevn thought it is not valid since this might be needed for debugging or optimization
}

void GoldenPatternResult::setStubResult(int layer, StubResult& stubResult) {
  if(stubResult.getValid() ) {
    //pdfSum += pdfVal;
    //firedLayerBits.set(layer);
    firedLayerBits |= (1<< layer);
  }
  stubResults[layer] = stubResult;

  //stub result is added evevn thought it is not valid since this might be needed for debugging or optimization
}

////////////////////////////////////////////
////////////////////////////////////////////
void GoldenPatternResult::init(const OMTFConfiguration* omtfConfig) {
  this->omtfConfig = omtfConfig;

  stubResults.assign(omtfConfig->nLayers(), StubResult() );
  reset();
}

void GoldenPatternResult::reset() {
  for(auto& stubResult : stubResults) {
    stubResult.reset();
  }
  valid = false;
  refLayer = -1;
  phi = 0;
  eta = 0;
  pdfSum = 0;
  firedLayerCnt = 0;
  firedLayerBits = 0;
  refHitPhi = 0;
  gpProbability1 = 0;
  gpProbability2 = 0;
}


/*void GoldenPatternResult::clear() {
  if(refLayerResults.size() == 0)
    refLayerResults.assign(myOmtfConfig->nRefLayers(), RefLayerResult());
  for (auto& reflayerRes: refLayerResults) {
    reflayerRes.reset();
  }
  results1D.assign(myOmtfConfig->nRefLayers(),0);
  hits1D.assign(myOmtfConfig->nRefLayers(),0);
  results.assign(myOmtfConfig->nLayers(),results1D);
  refPhi1D.assign(myOmtfConfig->nRefLayers(),1024);
  refEta1D.assign(myOmtfConfig->nRefLayers(),1024);
  hitsBits.assign(myOmtfConfig->nRefLayers(),0);  
  refPhiRHit1D.assign(myOmtfConfig->nRefLayers(),1024);
}*/
////////////////////////////////////////////
////////////////////////////////////////////
//default version
void GoldenPatternResult::finalise0() {
  for(unsigned int iLogicLayer=0; iLogicLayer < stubResults.size(); ++iLogicLayer) {
    unsigned int connectedLayer = omtfConfig->getLogicToLogic().at(iLogicLayer);
    //here we require that in case of the DT layers, both phi and phiB is fired
    if(firedLayerBits & (1<<connectedLayer) ) {
      if( firedLayerBits & (1<<iLogicLayer) ) {//now in the GoldenPattern::process1Layer1RefLayer the pdf bin 0 is returned when the layer is not fired, so this is 'if' is to assured that this pdf val is not added here
        pdfSum += stubResults[iLogicLayer].getPdfVal();

        if (omtfConfig->fwVersion() <= 4) {
          if(!omtfConfig->getBendingLayers().count(iLogicLayer)) //in DT case, the phi and phiB layers are threaded as one, so the firedLayerCnt is increased only for the phi layer
            firedLayerCnt++;
        }
        else
          firedLayerCnt++;
      }
    }
    else {
      firedLayerBits &= ~(1<<iLogicLayer);
    }
  }

  valid = true;
  //by default result becomes valid here, but can be overwritten later
}

////////////////////////////////////////////
////////////////////////////////////////////
//for the algo version with thresholds
void GoldenPatternResult::finalise1() {
  //cout<<__FUNCTION__<<":"<<__LINE__<<endl;
  for(unsigned int iLogicLayer=0; iLogicLayer < stubResults.size(); ++iLogicLayer) {
    //in this version we do not require that both phi and phiB is fired (non-zero), we thread them just independent
    //watch out that then the number of fired layers is bigger, and the cut on the minimal number of fired layers does not work in the same way as when the dt chamber is counted as one layer
    //TODO check if it affects performance
    pdfSum += stubResults[iLogicLayer].getPdfVal();
    firedLayerCnt += ( (firedLayerBits & (1<<iLogicLayer)) != 0 );
  }

  valid = true;
  //by default result becomes valid here, but can be overwritten later
}

////////////////////////////////////////////
////////////////////////////////////////////
//multiplication of PDF values instead of sum
void GoldenPatternResult::finalise2() {
  pdfSum = 1.;
  firedLayerCnt = 0;
  for(unsigned int iLogicLayer=0; iLogicLayer < stubResults.size(); ++iLogicLayer) {
    unsigned int connectedLayer = omtfConfig->getLogicToLogic().at(iLogicLayer);
    //here we require that in case of the DT layers, both phi and phiB is fired
    if(firedLayerBits & (1<<connectedLayer) ) {
      if( firedLayerBits & (1<<iLogicLayer) ) {//now in the GoldenPattern::process1Layer1RefLayer the pdf bin 0 is returned when the layer is not fired, so this is 'if' is to assured that this pdf val is not added here
        pdfSum *= stubResults[iLogicLayer].getPdfVal();
        if(!omtfConfig->getBendingLayers().count(iLogicLayer)) //in DT case, the phi and phiB layers are threaded as one, so the firedLayerCnt is increased only for the phi layer
          firedLayerCnt++;
      }
    }
    else {
      firedLayerBits &= ~(1<<iLogicLayer);
    }
  }

  if(firedLayerCnt < 3)
    pdfSum = 0;

  valid = true;
  //by default result becomes valid here, but can be overwritten later
}

////////////////////////////////////////////
////////////////////////////////////////////
//for patterns generation
void GoldenPatternResult::finalise3() {
  firedLayerCnt = 0;
  for(unsigned int iLogicLayer=0; iLogicLayer < stubResults.size(); ++iLogicLayer) {
    //in this version we do not require that both phi and phiB is fired (non-zero), we thread them just independent
    //watch out that then the number of fired layers is bigger, and the cut on the minimal number of fired layers dies not work in the same way as when the dt chamber is counted as one layer
    //TODO check if it affects performance
    pdfSum += stubResults[iLogicLayer].getPdfVal();

    if(stubResults[iLogicLayer].getPdfBin() != 5464 )//TODO in principle should (int)myOmtfConfig->nPhiBins(), but in GoldenPatternBase::process1Layer1RefLayer pdfMiddle is added
      firedLayerCnt ++;
  }

  valid = true;
  //by default result becomes valid here, but can be overwritten later
}

/*void GoldenPatternResult::finalise2() {
  pdfSum = 1.;
  for(unsigned int iLogicLayer=0; iLogicLayer < pdfValues.size(); ++iLogicLayer) {
    //in this version we do not require that both phi and phiB is fired (non-zero), we thread them just independent
    //TODO check if it affects performance
    bool layerFired = ( (firedLayerBits & (1<<iLogicLayer)) != 0 );
    if(layerFired) {
      pdfSum *= pdfValues[iLogicLayer];
      firedLayerCnt++ ;
    }
  }
  if(firedLayerCnt < 3)
    pdfSum = 0;
  valid = true;
  //by default result becomes valid here, but can be overwritten later
}*/

////////////////////////////////////////////
////////////////////////////////////////////
/*bool GoldenPatternResults::empty() const{

  unsigned int nHits = 0;
  for(unsigned int iRefLayer=0; iRefLayer<myOmtfConfig->nRefLayers(); ++iRefLayer){
    nHits+=hits1D[iRefLayer];
  }      
  return (nHits==0);
}*/
////////////////////////////////////////////
////////////////////////////////////////////
std::ostream & operator << (std::ostream &out, const GoldenPatternResult & gpResult) {
  unsigned int refLayerLogicNum = gpResult.omtfConfig->getRefToLogicNumber()[gpResult.getRefLayer()];

  for(unsigned int iLogicLayer=0; iLogicLayer < gpResult.stubResults.size(); ++iLogicLayer) {
    out<<" layer: "<<std::setw(2)<<iLogicLayer<<" hit: "
        <<std::setw(4)<<(gpResult.omtfConfig->isBendingLayer(iLogicLayer) ? gpResult.stubResults[iLogicLayer].getMuonStub()->phiBHw : gpResult.stubResults[iLogicLayer].getMuonStub()->phiHw )
        <<" pdfBin: "<<std::setw(4)<<gpResult.stubResults[iLogicLayer].getPdfBin()
        <<" pdfVal: "<<std::setw(3)<<gpResult.stubResults[iLogicLayer].getPdfVal()
       <<" fired "<<gpResult.isLayerFired(iLogicLayer)
       <<(iLogicLayer == refLayerLogicNum ? " <<< refLayer" : "")<<std::endl;
  }

  out<<"  refLayer: ";
  out << gpResult.getRefLayer()<<"\t";

  out<<" Sum over layers: ";
  out<<gpResult.getPdfSum()<<"\t";

  out<<" Number of hits: ";
  out << gpResult.getFiredLayerCnt()<<"\t";

  out<<" GpProbability1: ";
  out << gpResult.getGpProbability1()<<"\t";

  out<<" GpProbability2: ";
  out << gpResult.getGpProbability2()<<"\t";

  out<<std::endl;


  return out;
}
////////////////////////////////////////////
////////////////////////////////////////////
