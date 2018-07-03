#include "RecoCTPPS/PixelLocal/interface/RPixClusterToHit.h"


RPixClusterToHit::RPixClusterToHit(edm::ParameterSet const& conf)
{
verbosity_ = conf.getUntrackedParameter<int>("RPixVerbosity");
}

RPixClusterToHit::~RPixClusterToHit(){}

void RPixClusterToHit::buildHits(unsigned int detId, const std::vector<CTPPSPixelCluster> &clusters, std::vector<CTPPSPixelRecHit> &hits)
{
  
  if(verbosity_) edm::LogInfo("RPixClusterToHit")<<" RPixClusterToHit "<<detId<<" received cluster array of size = "<<clusters.size();
   for(unsigned int i=0; i<clusters.size();i++){
     make_hit(clusters[i],hits);
  }

}


void RPixClusterToHit::make_hit(CTPPSPixelCluster aCluster,  std::vector<CTPPSPixelRecHit> &hits ){

// take a cluster, generate a rec hit and push it in the rec hit vector

//call the topology
  CTPPSPixelSimTopology topology;
//call the numbering inside the ROC
  CTPPSPixelIndices pxlInd;
// get information from the cluster 
// get the whole cluster size and row/col size
  unsigned int thisClusterSize = aCluster.size();
  unsigned int thisClusterRowSize = aCluster.sizeRow();
  unsigned int thisClusterColSize = aCluster.sizeCol();

// get the minimum pixel row/col
  unsigned int thisClusterMinRow = aCluster.minPixelRow();
  unsigned int thisClusterMinCol = aCluster.minPixelCol();

// calculate "on edge" flag
  bool anEdgePixel = false;
  if(aCluster.minPixelRow() == 0 || aCluster.minPixelCol() == 0 ||  
     int(aCluster.minPixelRow()+aCluster.rowSpan()) == (pxlInd.getDefaultRowDetSize()-1) ||  
     int(aCluster.minPixelCol()+aCluster.colSpan()) == (pxlInd.getDefaultColDetSize()-1) ) 
    anEdgePixel = true;

// check for bad (ADC=0) pixels in cluster 
  bool aBadPixel = false;
  for(unsigned int i = 0; i < thisClusterSize; i++){
    if(aCluster.pixelADC(i)==0) aBadPixel = true;
  } 

// check for spanning two ROCs
  bool twoRocs = false;
  int currROCId = pxlInd.getROCId(aCluster.pixelCol(0),aCluster.pixelRow(0) ); 

  for(unsigned int i = 1; i < thisClusterSize; i++){
    if(pxlInd.getROCId(aCluster.pixelCol(i),aCluster.pixelRow(i) ) != currROCId){
      twoRocs = true;
      break;
    }
  }

//estimate position and error of the hit
  double avgWLocalX = 0;
  double avgWLocalY = 0;
  double weights = 0;
  double weightedVarianceX = 0.;
  double weightedVarianceY = 0.;

  if(verbosity_)
    edm::LogInfo("RPixClusterToHit") << " hit pixels: ";

  for(unsigned int i = 0; i < thisClusterSize; i++){
    
    if(verbosity_)edm::LogInfo("RPixClusterToHit") << aCluster.pixelRow(i) << " " << aCluster.pixelCol(i)<<" " << aCluster.pixelADC(i);

    double minPxlX = 0;
    double minPxlY = 0;
    double maxPxlX = 0;
    double maxPxlY = 0;
    topology.pixelRange(aCluster.pixelRow(i),aCluster.pixelCol(i),minPxlX,maxPxlX,minPxlY, maxPxlY);
    double halfSizeX = (maxPxlX-minPxlX)/2.;
    double halfSizeY = (maxPxlY-minPxlY)/2.;
    double avgPxlX = minPxlX + halfSizeX;
    double avgPxlY = minPxlY + halfSizeY;
//error propagation
    weightedVarianceX += aCluster.pixelADC(i)*aCluster.pixelADC(i)*halfSizeX*halfSizeX/3.;
    weightedVarianceY += aCluster.pixelADC(i)*aCluster.pixelADC(i)*halfSizeY*halfSizeY/3.;
    
    avgWLocalX += avgPxlX*aCluster.pixelADC(i);
    avgWLocalY += avgPxlY*aCluster.pixelADC(i);
    weights += aCluster.pixelADC(i);

  } 

  if(weights == 0){
    edm::LogError("RPixClusterToHit") << " unexpected weights = 0 for cluster (Row_min, Row_max, Col_min, Col_max) = ("
				      << aCluster.minPixelRow() << ","
				      << aCluster.minPixelRow()+aCluster.rowSpan() << ","
				      << aCluster.minPixelCol() << ","
				      << aCluster.minPixelCol()+aCluster.colSpan()
				      << ")"; 
    return;
  }

  double invWeights = 1./weights;
  double avgLocalX = avgWLocalX*invWeights;
  double avgLocalY = avgWLocalY*invWeights;
  
  double varianceX = weightedVarianceX*invWeights*invWeights;
  double varianceY = weightedVarianceY*invWeights*invWeights;

  LocalPoint lp(avgLocalX,avgLocalY,0);
  LocalError le(varianceX,0,varianceY);
  CTPPSPixelRecHit rh(lp,le,anEdgePixel,aBadPixel,twoRocs,thisClusterMinRow,thisClusterMinCol,thisClusterSize,thisClusterRowSize,thisClusterColSize);
  if(verbosity_)edm::LogInfo("RPixClusterToHit") << lp << " with error " << le ;

  hits.push_back(rh);

  return;

}





