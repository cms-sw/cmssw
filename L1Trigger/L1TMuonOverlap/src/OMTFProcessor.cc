#include <iostream>
#include <algorithm>
#include <strstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"

#include "L1Trigger/L1TMuonOverlap/interface/OMTFProcessor.h"
#include "L1Trigger/L1TMuonOverlap/interface/GoldenPattern.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFinput.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFResult.h"

#include "L1Trigger/RPCTrigger/interface/RPCConst.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
///////////////////////////////////////////////
///////////////////////////////////////////////
OMTFProcessor::~OMTFProcessor(){

   for(auto it: theGPs) delete it.second;
   
}
///////////////////////////////////////////////
///////////////////////////////////////////////
void OMTFProcessor::resetConfiguration(){

  myResults.clear();
  theGPs.clear();
}
///////////////////////////////////////////////
///////////////////////////////////////////////
bool OMTFProcessor::configure(const OMTFConfiguration * omtfConfig,
			      const L1TMuonOverlapParams * omtfPatterns){
			      
  resetConfiguration();

  myOmtfConfig = omtfConfig;
  
  myResults.assign(myOmtfConfig->nTestRefHits(),OMTFProcessor::resultsMap());
  
  const l1t::LUT* chargeLUT =  omtfPatterns->chargeLUT();
  const l1t::LUT* etaLUT =  omtfPatterns->etaLUT();
  const l1t::LUT* ptLUT =  omtfPatterns->ptLUT();
  const l1t::LUT* pdfLUT =  omtfPatterns->pdfLUT();
  const l1t::LUT* meanDistPhiLUT =  omtfPatterns->meanDistPhiLUT();

  unsigned int nGPs = myOmtfConfig->nGoldenPatterns();
  unsigned int address = 0;
  unsigned int iEta, iPt;
  int iCharge;
  for(unsigned int iGP=0;iGP<nGPs;++iGP){
    address = iGP;
    iEta = etaLUT->data(address);
    iCharge = chargeLUT->data(address)==0? -1:1;
    iPt = ptLUT->data(address);

    GoldenPattern::vector2D meanDistPhi2D(myOmtfConfig->nLayers());
    GoldenPattern::vector1D pdf1D(exp2(myOmtfConfig->nPdfAddrBits()));
    GoldenPattern::vector3D pdf3D(myOmtfConfig->nLayers());
    GoldenPattern::vector2D pdf2D(myOmtfConfig->nRefLayers());
    ///Mean dist phi data
    for(unsigned int iLayer=0;iLayer<myOmtfConfig->nLayers();++iLayer){
      GoldenPattern::vector1D meanDistPhi1D(myOmtfConfig->nRefLayers());
      for(unsigned int iRefLayer=0;iRefLayer<myOmtfConfig->nRefLayers();++iRefLayer){
	address = iRefLayer + iLayer*myOmtfConfig->nRefLayers() + iGP*(myOmtfConfig->nRefLayers()*myOmtfConfig->nLayers());
	meanDistPhi1D[iRefLayer] = meanDistPhiLUT->data(address) - (1<<(meanDistPhiLUT->nrBitsData() -1));	
      }
      meanDistPhi2D[iLayer] = meanDistPhi1D;    
      ///Pdf data
      for(unsigned int iRefLayer=0;iRefLayer<myOmtfConfig->nRefLayers();++iRefLayer){
	pdf1D.assign(1<<myOmtfConfig->nPdfAddrBits(),0);
	for(unsigned int iPdf=0;iPdf<(unsigned int)(1<<myOmtfConfig->nPdfAddrBits());++iPdf){
	  address = iPdf + iRefLayer*(1<<myOmtfConfig->nPdfAddrBits()) +
	    iLayer*myOmtfConfig->nRefLayers()*(1<<myOmtfConfig->nPdfAddrBits()) +
	    iGP*myOmtfConfig->nLayers()*myOmtfConfig->nRefLayers()*(1<<myOmtfConfig->nPdfAddrBits());
	  pdf1D[iPdf] = pdfLUT->data(address);
	}
	pdf2D[iRefLayer] = pdf1D;
      }
      pdf3D[iLayer] = pdf2D;
    }
    Key aKey(iEta,iPt,iCharge);

    GoldenPattern *aGP = new GoldenPattern(aKey, myOmtfConfig);
    aGP->setMeanDistPhi(meanDistPhi2D);
    aGP->setPdf(pdf3D);
    addGP(aGP);    
  }
  return true;
}
///////////////////////////////////////////////
///////////////////////////////////////////////
bool OMTFProcessor::addGP(GoldenPattern *aGP){

  if(theGPs.find(aGP->key())!=theGPs.end()){
    throw cms::Exception("Corrupted Golden Patterns data")
      <<"OMTFProcessor::addGP(...) "
      <<" Reading two Golden Patterns with the same key: "
      <<aGP->key()<<std::endl;
  }
  else theGPs[aGP->key()] = new GoldenPattern(*aGP);

  for(auto & itRegion: myResults){
    OMTFResult aResult;
    aResult.configure(myOmtfConfig);    
    itRegion[aGP->key()] = aResult;
  }

  return true;
}
///////////////////////////////////////////////
///////////////////////////////////////////////
void  OMTFProcessor::averagePatterns(int charge){

  Key aKey(0, 9, charge);

  while(theGPs.find(aKey)!=theGPs.end()){

    GoldenPattern *aGP1 = theGPs.find(aKey)->second;
    GoldenPattern *aGP2 = aGP1;
    GoldenPattern *aGP3 = aGP1;
    GoldenPattern *aGP4 = aGP1;

    ++aKey.thePtCode;
    while(theGPs.find(aKey)==theGPs.end() && aKey.thePtCode<=401) ++aKey.thePtCode;    
    if(aKey.thePtCode<=401 && theGPs.find(aKey)!=theGPs.end()) aGP2 =  theGPs.find(aKey)->second;
    
    if(aKey.thePtCode>71){
      ++aKey.thePtCode;
      while(theGPs.find(aKey)==theGPs.end() && aKey.thePtCode<=401) ++aKey.thePtCode;    
      if(aKey.thePtCode<=401 && theGPs.find(aKey)!=theGPs.end()) aGP3 =  theGPs.find(aKey)->second;

      ++aKey.thePtCode;
      while(theGPs.find(aKey)==theGPs.end() && aKey.thePtCode<=401) ++aKey.thePtCode;    
      if(aKey.thePtCode<=401 && theGPs.find(aKey)!=theGPs.end()) aGP4 =  theGPs.find(aKey)->second;
    }
    else{
      aGP3 = aGP1;
      aGP4 = aGP2;
    }    
    //HACK. Have to clean this up.
    ///Previously pt codes were going by steps of 1, now this is not the case
    ++aKey.thePtCode;
    while(theGPs.find(aKey)==theGPs.end() && aKey.thePtCode<=401) ++aKey.thePtCode;    
    ///////////////////////////////
    
    
    GoldenPattern::vector2D meanDistPhi  = aGP1->getMeanDistPhi();

    GoldenPattern::vector2D meanDistPhi1  = aGP1->getMeanDistPhi();
    GoldenPattern::vector2D meanDistPhi2  = aGP2->getMeanDistPhi();
    GoldenPattern::vector2D meanDistPhi3  = aGP3->getMeanDistPhi();
    GoldenPattern::vector2D meanDistPhi4  = aGP4->getMeanDistPhi();
   
    for(unsigned int iLayer=0;iLayer<myOmtfConfig->nLayers();++iLayer){
      for(unsigned int iRefLayer=0;iRefLayer<myOmtfConfig->nRefLayers();++iRefLayer){
      	meanDistPhi[iLayer][iRefLayer]+=meanDistPhi2[iLayer][iRefLayer];
      	meanDistPhi[iLayer][iRefLayer]+=meanDistPhi3[iLayer][iRefLayer];
      	meanDistPhi[iLayer][iRefLayer]+=meanDistPhi4[iLayer][iRefLayer];
      	meanDistPhi[iLayer][iRefLayer]/=4;
      }
    }
    
    aGP1->setMeanDistPhi(meanDistPhi);
    aGP2->setMeanDistPhi(meanDistPhi);

    shiftGP(aGP1,meanDistPhi, meanDistPhi1);
    shiftGP(aGP2,meanDistPhi, meanDistPhi2);   
    if(aGP3!=aGP1 && aGP4!=aGP2){
      aGP3->setMeanDistPhi(meanDistPhi);
      aGP4->setMeanDistPhi(meanDistPhi);
      shiftGP(aGP3,meanDistPhi, meanDistPhi3);   
      shiftGP(aGP4,meanDistPhi, meanDistPhi4);   
    }
  }
}
///////////////////////////////////////////////
///////////////////////////////////////////////
void OMTFProcessor::shiftGP(GoldenPattern *aGP,
			    const GoldenPattern::vector2D & meanDistPhiNew,
			    const GoldenPattern::vector2D & meanDistPhiOld){

  ///Shift pdfs by differecne between original menaDistPhi, and
  ///the averaged value
  unsigned int nPdfBins =  exp2(myOmtfConfig->nPdfAddrBits());
  GoldenPattern::vector3D pdfAllRef = aGP->getPdf();

  int indexShift = 0;
  for(unsigned int iLayer=0;iLayer<myOmtfConfig->nLayers();++iLayer){
    for(unsigned int iRefLayer=0;iRefLayer<myOmtfConfig->nRefLayers();++iRefLayer){
      indexShift = meanDistPhiOld[iLayer][iRefLayer] - meanDistPhiNew[iLayer][iRefLayer];
      for(unsigned int iPdfBin=0;iPdfBin<nPdfBins;++iPdfBin) pdfAllRef[iLayer][iRefLayer][iPdfBin] = 0;
	for(unsigned int iPdfBin=0;iPdfBin<nPdfBins;++iPdfBin){
	  if((int)(iPdfBin)+indexShift>=0 && iPdfBin+indexShift<nPdfBins)
	    pdfAllRef[iLayer][iRefLayer][iPdfBin+indexShift] = aGP->pdfValue(iLayer, iRefLayer, iPdfBin);
	}
      }
    }
    aGP->setPdf(pdfAllRef);
}
///////////////////////////////////////////////
///////////////////////////////////////////////
const std::map<Key,GoldenPattern*> & OMTFProcessor::getPatterns() const{ return theGPs; }
///////////////////////////////////////////////
///////////////////////////////////////////////
const std::vector<OMTFProcessor::resultsMap> & OMTFProcessor::processInput(unsigned int iProcessor,
									   const OMTFinput & aInput){

  for(auto & itRegion: myResults) for(auto & itKey: itRegion) itKey.second.clear();

  //////////////////////////////////////
  //////////////////////////////////////  
  std::bitset<128> refHitsBits = aInput.getRefHits(iProcessor);
  if(refHitsBits.none()) return myResults;
   
  for(unsigned int iLayer=0;iLayer<myOmtfConfig->nLayers();++iLayer){
    const OMTFinput::vector1D & layerHits = aInput.getLayerData(iLayer);
    if(!layerHits.size()) continue;
    ///Number of reference hits to be checked. 
    unsigned int nTestedRefHits = myOmtfConfig->nTestRefHits();
    for(unsigned int iRefHit=0;iRefHit<myOmtfConfig->nRefHits();++iRefHit){
      if(!refHitsBits[iRefHit]) continue;
      if(nTestedRefHits--==0) break;
      const RefHitDef & aRefHitDef = myOmtfConfig->getRefHitsDefs()[iProcessor][iRefHit];
      
      int phiRef = aInput.getLayerData(myOmtfConfig->getRefToLogicNumber()[aRefHitDef.iRefLayer])[aRefHitDef.iInput];
      int etaRef = aInput.getLayerData(myOmtfConfig->getRefToLogicNumber()[aRefHitDef.iRefLayer],true)[aRefHitDef.iInput];
      unsigned int iRegion = aRefHitDef.iRegion;
      
      if(myOmtfConfig->getBendingLayers().count(iLayer)) phiRef = 0;
      const OMTFinput::vector1D restrictedLayerHits = restrictInput(iProcessor, iRegion, iLayer,layerHits);
      for(auto itGP: theGPs){
      	GoldenPattern::layerResult aLayerResult = itGP.second->process1Layer1RefLayer(aRefHitDef.iRefLayer,iLayer,
      										      phiRef,
      										      restrictedLayerHits);             
      	int phiRefSt2 = itGP.second->propagateRefPhi(phiRef, etaRef, aRefHitDef.iRefLayer);       	
      	myResults[myOmtfConfig->nTestRefHits()-nTestedRefHits-1][itGP.second->key()].setRefPhiRHits(aRefHitDef.iRefLayer, phiRef); 
        myResults[myOmtfConfig->nTestRefHits()-nTestedRefHits-1][itGP.second->key()].addResult(aRefHitDef.iRefLayer,iLayer,
													      aLayerResult.first,
													      phiRefSt2,etaRef);	 
      }
    }
  }
  //////////////////////////////////////
  ////////////////////////////////////// 
  for(auto & itRefHit: myResults) for(auto & itKey: itRefHit) itKey.second.finalise();

  std::ostringstream myStr;
  myStr<<"iProcessor: "<<iProcessor<<std::endl;
  myStr<<"Input: ------------"<<std::endl;
  myStr<<aInput<<std::endl; 
  edm::LogInfo("OMTF processor")<<myStr.str();
  
  return myResults;
}   
////////////////////////////////////////////
////////////////////////////////////////////
OMTFinput::vector1D OMTFProcessor::restrictInput(unsigned int iProcessor,
						 unsigned int iRegion,
						 unsigned int iLayer,
						 const OMTFinput::vector1D & layerHits){

  OMTFinput::vector1D myHits = layerHits;
  
  unsigned int iStart = myOmtfConfig->getConnections()[iProcessor][iRegion][iLayer].first;
  unsigned int iEnd = iStart + myOmtfConfig->getConnections()[iProcessor][iRegion][iLayer].second -1;

  for(unsigned int iInput=0;iInput<14;++iInput){    
    if(iInput<iStart || iInput>iEnd) myHits[iInput] = myOmtfConfig->nPhiBins();
  }  
  return myHits;
}
////////////////////////////////////////////
////////////////////////////////////////////
void OMTFProcessor::fillCounts(unsigned int iProcessor,
			       const OMTFinput & aInput,
			       const SimTrack* aSimMuon){

  int theCharge = abs(aSimMuon->type()) == 13 ? -1 : 1; 
  unsigned int  iPt =  RPCConst::iptFromPt(aSimMuon->momentum().pt());
  ///Stupid conersion. Have to go through PAC pt scale, as we later
  ///shift resulting pt code by +1
  iPt+=1;
  if(iPt>31) iPt=200*2+1;
  else iPt = RPCConst::ptFromIpt(iPt)*2.0+1;//MicroGMT has 0.5 GeV step size, with lower bin edge  (uGMT_pt_code - 1)*step_size
  //////

  //////////////////////////////////////  
  std::bitset<128> refHitsBits = aInput.getRefHits(iProcessor);
  if(refHitsBits.none()) return;

  std::ostringstream myStr;
  myStr<<"iProcessor: "<<iProcessor<<std::endl;
  myStr<<"Input: ------------"<<std::endl;
  myStr<<aInput<<std::endl;
  edm::LogInfo("OMTF processor")<<myStr.str();
   
  for(unsigned int iLayer=0;iLayer<myOmtfConfig->nLayers();++iLayer){
    const OMTFinput::vector1D & layerHits = aInput.getLayerData(iLayer);
    if(!layerHits.size()) continue;
    ///Number of reference hits to be checked. 
    for(unsigned int iRefHit=0;iRefHit<myOmtfConfig->nRefHits();++iRefHit){
      if(!refHitsBits[iRefHit]) continue;
      const RefHitDef & aRefHitDef = myOmtfConfig->getRefHitsDefs()[iProcessor][iRefHit];
      int phiRef = aInput.getLayerData(myOmtfConfig->getRefToLogicNumber()[aRefHitDef.iRefLayer])[aRefHitDef.iInput]; 
      unsigned int iRegion = aRefHitDef.iRegion;
      if(myOmtfConfig->getBendingLayers().count(iLayer)) phiRef = 0;
      const OMTFinput::vector1D restrictedLayerHits = restrictInput(iProcessor, iRegion, iLayer,layerHits);
      for(auto itGP: theGPs){	
	if(itGP.first.theCharge!=theCharge) continue;
	if(itGP.first.thePtCode!=iPt) continue;
        itGP.second->addCount(aRefHitDef.iRefLayer,iLayer,phiRef,restrictedLayerHits);
      }
    }
  }
}
////////////////////////////////////////////
////////////////////////////////////////////
