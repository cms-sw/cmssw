#ifndef RecoPixelVertexing_FindPeakFastPV_h
#define RecoPixelVertexing_FindPeakFastPV_h
/** \class FindPeakFastPV FindPeakFastPV.h RecoPixelVertexing/PixelVertexFinding/FindPeakFastPV.h 
 * Given *zProjections* and *zWeights* find the peak of width *m_zClusterWidth*. 
 * Use only values with *zWeights*>*m_weightCut*. 
 * Look near *oldVertex* within *m_zClusterSearchArea*.
 *
 *  $Date: 2013/12/17 17:33:24 $
 *  $Revision: 1.0 $
 *  \author Silvio Donato (SNS)
 */
//#include "FWCore/Framework/interface/EDFilter.h"
//#include "FWCore/Framework/interface/Frameworkfwd.h"
//#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/EventSetup.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include <map>
//#include <set>


float FindPeakFastPV(  std::vector<float> &zProjections, std::vector<float> &zWeights, float oldVertex, const float m_zClusterWidth, const float m_zClusterSearchArea, const float m_weightCut){
float centerWMax= oldVertex;
if( m_zClusterWidth > 0 && m_zClusterSearchArea >0 )
{
  std::vector<float>::iterator itCenter = zProjections.begin();
  std::vector<float>::iterator itLeftSide = zProjections.begin();
  std::vector<float>::iterator itRightSide = zProjections.begin();
  float zClusterWidth = m_zClusterWidth/2.0; //take half zCluster width 
  float maxW=0;
  for(;itCenter!=zProjections.end(); itCenter++)
  {

    if( (*itCenter < (oldVertex - zClusterWidth - m_zClusterSearchArea  ) ) || (*itCenter > (oldVertex + zClusterWidth + m_zClusterSearchArea ) )) continue; //loop only on the zProjections within oldVertex +/- (zClusterWidth + m_zClusterSearchArea)
    while(itLeftSide != zProjections.end() && (*itCenter - *itLeftSide) > zClusterWidth  ) itLeftSide++;
    while(itRightSide != zProjections.end() && (*itRightSide - *itCenter) < zClusterWidth  ) itRightSide++;
    float nWeighted = 0;
    float centerW= 0;
    
    for(std::vector<float>::iterator ii = itLeftSide; ii != itRightSide; ii++) 
    {//loop inside the peak and calculate its weight
     if(zWeights[ii-zProjections.begin()]<m_weightCut) continue;
     nWeighted+=zWeights[ii-zProjections.begin()]; 
     centerW+=(*ii)*zWeights[ii-zProjections.begin()]; 
    }
    centerW/=nWeighted; //calculate the weighted peak center
    if(nWeighted > maxW)
    {
       maxW=nWeighted;
       centerWMax=centerW;	       
    }
  }  
}
return centerWMax;
}

#endif
