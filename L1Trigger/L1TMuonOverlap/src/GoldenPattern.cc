#include <iostream>
#include <iomanip>
#include <cmath>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuonOverlap/interface/GoldenPattern.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFinput.h"

////////////////////////////////////////////////////
////////////////////////////////////////////////////
GoldenPattern::layerResult GoldenPattern::process1Layer1RefLayer(unsigned int iRefLayer,
								 unsigned int iLayer,
								 const int phiRefHit,
								 const OMTFinput::vector1D & layerHits){

  GoldenPattern::layerResult aResult;

  int phiMean = meanDistPhi[iLayer][iRefLayer];
  int phiDist = exp2(myOmtfConfig->nPdfAddrBits());
  ///Select hit closest to the mean of probability 
  ///distribution in given layer
  for(auto itHit: layerHits){
    if(itHit>=(int)myOmtfConfig->nPhiBins()) continue;
    if(abs(itHit-phiMean-phiRefHit)<abs(phiDist)) phiDist = itHit-phiMean-phiRefHit;
  }

  ///Shift phidist, so 0 is at the middle of the range
  phiDist+=exp2(myOmtfConfig->nPdfAddrBits()-1);
  ///Check if phiDist is within pdf range
  ///in -64 +63 U2 code
  ///Find more elegant way to check this.
  if(phiDist<0 ||
     phiDist>exp2(myOmtfConfig->nPdfAddrBits())-1){
    return aResult;
  }
  
  int pdfVal = pdfAllRef[iLayer][iRefLayer][phiDist];
  return GoldenPattern::layerResult(pdfVal,pdfVal>0);
}
////////////////////////////////////////////////////
////////////////////////////////////////////////////
int GoldenPattern::propagateRefPhi(int phiRef, int etaRef, unsigned int iRefLayer){

  unsigned int iLayer = 2; //MB2 
  //if(etaRef>101) iLayer = 7;//RE2
  return phiRef + meanDistPhi[iLayer][iRefLayer];
  
}
////////////////////////////////////////////////////
////////////////////////////////////////////////////
void GoldenPattern::addCount(unsigned int iRefLayer,
			     unsigned int iLayer,
			     const int phiRefHit,
			     const OMTFinput::vector1D & layerHits){

  int nHitsInLayer = 0;
  int phiDist = exp2(myOmtfConfig->nPdfAddrBits());
  for(auto itHit: layerHits){
    if(itHit>=(int)myOmtfConfig->nPhiBins()) continue;
    if(abs(itHit-phiRefHit)<phiDist) phiDist = itHit-phiRefHit;
    ++nHitsInLayer;
  }
  ///For making the patterns take events with a single hit in each layer
  if(nHitsInLayer>1 || nHitsInLayer==0) return;

  ///Shift phiDist so it is in +-Pi range
  if(phiDist>=(int)myOmtfConfig->nPhiBins()/2) phiDist-=(int)myOmtfConfig->nPhiBins();
  if(phiDist<=-(int)myOmtfConfig->nPhiBins()/2) phiDist+=(int)myOmtfConfig->nPhiBins();
  
  ///Shift phidist, so 0 is at the middle of the range
  int phiDistShift=phiDist+exp2(myOmtfConfig->nPdfAddrBits()-1);
  
  ///Check if phiDist is within pdf range
  ///in -64 +63 U2 code
  ///Find more elegant way to check this.
  if(phiDistShift<0 ||
     phiDistShift>exp2(myOmtfConfig->nPdfAddrBits())-1){
    return;
  }

  if((int)iLayer==myOmtfConfig->getRefToLogicNumber()[iRefLayer]) ++meanDistPhiCounts[iLayer][iRefLayer];
  ++pdfAllRef[iLayer][iRefLayer][phiDistShift];
}
////////////////////////////////////////////////////
////////////////////////////////////////////////////
std::ostream & operator << (std::ostream &out, const GoldenPattern & aPattern){

  out <<"GoldenPattern "<< aPattern.theKey <<std::endl;
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
/*
  out<<"PDF per layer:"<<std::endl;
  for (unsigned int iRefLayer=0;iRefLayer<aPattern.pdfAllRef[0].size();++iRefLayer){
    out<<"Ref layer: "<<iRefLayer;
    for (unsigned int iLayer=0;iLayer<aPattern.pdfAllRef.size();++iLayer){   
      out<<", measurement layer: "<<iLayer<<std::endl;
      for (unsigned int iPdf=0;iPdf<exp2(myOmtfConfig->nPdfAddrBits());++iPdf){   
	out<<std::setw(2)<<aPattern.pdfAllRef[iLayer][iRefLayer][iPdf]<<" ";
      }
      out<<std::endl;
    }
  }
*/
  return out;
}
////////////////////////////////////////////////////
////////////////////////////////////////////////////
void GoldenPattern::reset(){

  GoldenPattern::vector1D meanDistPhi1D(myOmtfConfig->nRefLayers());
  GoldenPattern::vector2D meanDistPhi2D(myOmtfConfig->nLayers());
  meanDistPhi2D.assign(myOmtfConfig->nLayers(), meanDistPhi1D);
  meanDistPhi = meanDistPhi2D;
  meanDistPhiCounts = meanDistPhi2D;

  GoldenPattern::vector1D pdf1D(exp2(myOmtfConfig->nPdfAddrBits()));
  GoldenPattern::vector2D pdf2D(myOmtfConfig->nRefLayers());
  GoldenPattern::vector3D pdf3D(myOmtfConfig->nLayers());

  pdf2D.assign(myOmtfConfig->nRefLayers(),pdf1D);
  pdfAllRef.assign(myOmtfConfig->nLayers(),pdf2D);
}
////////////////////////////////////////////////////
////////////////////////////////////////////////////
void GoldenPattern::normalise(){

  for (unsigned int iRefLayer=0;iRefLayer<pdfAllRef[0].size();++iRefLayer){
    for (unsigned int iLayer=0;iLayer<pdfAllRef.size();++iLayer){   
      for (unsigned int iPdf=0;iPdf<pdfAllRef[iLayer][iRefLayer].size();++iPdf){   
	float pVal = log((float)pdfAllRef[iLayer][iRefLayer][iPdf]/meanDistPhiCounts[myOmtfConfig->getRefToLogicNumber()[iRefLayer]][iRefLayer]);
	if(pVal<log(myOmtfConfig->minPdfVal())) continue;
	meanDistPhi[iLayer][iRefLayer]+=(iPdf - exp2(myOmtfConfig->nPdfAddrBits()-1))*pdfAllRef[iLayer][iRefLayer][iPdf];
	if((int)iLayer!=myOmtfConfig->getRefToLogicNumber()[iRefLayer]) meanDistPhiCounts[iLayer][iRefLayer]+=pdfAllRef[iLayer][iRefLayer][iPdf];
      }
    }
  }

  ///Mean dist phi  
  for (unsigned int iRefLayer=0;iRefLayer<meanDistPhi[0].size();++iRefLayer){
    for (unsigned int iLayer=0;iLayer<meanDistPhi.size();++iLayer){   
      if(meanDistPhiCounts.size() && meanDistPhiCounts[iLayer][iRefLayer]){
	if(meanDistPhiCounts[iLayer][iRefLayer]<1000) 	meanDistPhi[iLayer][iRefLayer] = 0;
	else meanDistPhi[iLayer][iRefLayer] = rint((float)meanDistPhi[iLayer][iRefLayer]/meanDistPhiCounts[iLayer][iRefLayer]);      
      }
    }
  }  
  const float minPlog =  log(myOmtfConfig->minPdfVal());
  const unsigned int nPdfValBits = myOmtfConfig->nPdfValBits(); 
  ///Probabilities. Normalise and change from float to integer values
  float pVal;
  int digitisedVal, truncatedValue;
  for (unsigned int iRefLayer=0;iRefLayer<pdfAllRef[0].size();++iRefLayer){
    for (unsigned int iLayer=0;iLayer<pdfAllRef.size();++iLayer){   
      for (unsigned int iPdf=0;iPdf<pdfAllRef[iLayer][iRefLayer].size();++iPdf){   
	if(!meanDistPhiCounts[myOmtfConfig->getRefToLogicNumber()[iRefLayer]][iRefLayer] ||
	   !pdfAllRef[iLayer][iRefLayer][iPdf]) continue;	
	pVal = log((float)pdfAllRef[iLayer][iRefLayer][iPdf]/meanDistPhiCounts[myOmtfConfig->getRefToLogicNumber()[iRefLayer]][iRefLayer]);
	///If there are only a few counts in given measurement layer, set pdf value to 0
	if(pVal<minPlog || meanDistPhiCounts[iLayer][iRefLayer]<1000){
	  pdfAllRef[iLayer][iRefLayer][iPdf] = 0;
	  continue;
	}
	///Digitisation
	///Values remapped 0->std::pow(2,nPdfValBits)
	///          minPlog->0 
	digitisedVal = rint((std::pow(2,nPdfValBits)-1) - (pVal/minPlog)*(std::pow(2,nPdfValBits)-1));
	///Make sure digitised value is saved using nBitsVal bits
	truncatedValue  = 0 | (digitisedVal  & ((int)pow(2,nPdfValBits)-1));
	pdfAllRef[iLayer][iRefLayer][iPdf] = truncatedValue;
      }
    }
  } 

       
  vector3D  pdfAllRefTmp = pdfAllRef;

  const unsigned int nPdfAddrBits = 7;
  for (unsigned int iRefLayer=0;iRefLayer<pdfAllRef[0].size();++iRefLayer){
    for (unsigned int iLayer=0;iLayer<pdfAllRef.size();++iLayer){   
      for (unsigned int iPdf=0;iPdf<pdfAllRef[iLayer][iRefLayer].size();++iPdf){
	pdfAllRef[iLayer][iRefLayer][iPdf] = 0;   
	///Shift pdf index by meanDistPhi
	int index = iPdf - exp2(myOmtfConfig->nPdfAddrBits()-1)  - meanDistPhi[iLayer][iRefLayer] + exp2(nPdfAddrBits-1);       
	if(index<0 || index>exp2(nPdfAddrBits)-1) continue;
	pdfAllRef[iLayer][iRefLayer][index] = pdfAllRefTmp[iLayer][iRefLayer][iPdf];
      }
    }
  }
}
////////////////////////////////////////////////////
////////////////////////////////////////////////////
bool GoldenPattern::hasCounts(){

 for (unsigned int iRefLayer=0;iRefLayer<meanDistPhi[0].size();++iRefLayer){
    for (unsigned int iLayer=0;iLayer<meanDistPhi.size();++iLayer){   
      if(meanDistPhiCounts.size() && meanDistPhiCounts[iLayer][iRefLayer]) return true;
  }
 }
 return false;
}
////////////////////////////////////////////////////
////////////////////////////////////////////////////
