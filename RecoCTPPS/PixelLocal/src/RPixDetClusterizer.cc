#include <iostream>
#include "FWCore/Utilities/interface/Exception.h"

#include "RecoCTPPS/PixelLocal/interface/RPixDetClusterizer.h"

namespace {
  constexpr int max16bits = 65535;
  constexpr int maxCol = 155;
  constexpr int maxRow = 159;
  constexpr double highRangeCal = 1800.;
  constexpr double lowRangeCal = 260.;
}


RPixDetClusterizer::RPixDetClusterizer(edm::ParameterSet const& conf):
  params_(conf), SeedVector_(0)
{
  verbosity_ = conf.getUntrackedParameter<int>("RPixVerbosity");
  SeedADCThreshold_ = conf.getParameter<int>("SeedADCThreshold");
  ADCThreshold_ = conf.getParameter<int>("ADCThreshold");
  ElectronADCGain_ = conf.getParameter<double>("ElectronADCGain");
  VcaltoElectronGain_ = conf.getParameter<int>("VCaltoElectronGain");
  VcaltoElectronOffset_ = conf.getParameter<int>("VCaltoElectronOffset");
  doSingleCalibration_ = conf.getParameter<bool>("doSingleCalibration");

}

RPixDetClusterizer::~RPixDetClusterizer(){}

void RPixDetClusterizer::buildClusters(unsigned int detId, const std::vector<CTPPSPixelDigi> &digi, std::vector<CTPPSPixelCluster> &clusters, const CTPPSPixelGainCalibrations * pcalibrations,const CTPPSPixelAnalysisMask* maskera)
{

  std::map<uint32_t, CTPPSPixelROCAnalysisMask> mask = maskera->analysisMask;
  std::map<uint32_t, CTPPSPixelROCAnalysisMask>::iterator mask_it = mask.find(detId); 

  std::set<std::pair<unsigned char, unsigned char> > maskedPixels;
  if( mask_it != mask.end()) maskedPixels = mask_it->second.maskedPixels;

  if(verbosity_) edm::LogInfo("RPixDetClusterizer")<<detId<<" received digi.size()="<<digi.size();

// creating a set of CTPPSPixelDigi's and fill it

  rpix_digi_set_.clear();
  rpix_digi_set_.insert(digi.begin(),digi.end());

// calibrate digis here
  calib_rpix_digi_set_.clear();

  for( std::set<CTPPSPixelDigi>::iterator RPdit = rpix_digi_set_.begin(); RPdit != rpix_digi_set_.end();++RPdit){

    if((*RPdit).row() > maxRow || (*RPdit).column() > maxCol)
      throw cms::Exception("CTPPSDigiOutofRange") 
	<< " row = " << (*RPdit).row() << "  column = "<< (*RPdit).column();

    unsigned char row = (*RPdit).row();
    unsigned char column = (*RPdit).column();
    unsigned short adc = (*RPdit).adc();
    unsigned short electrons = 0;
    std::pair<unsigned char, unsigned char> pixel = std::make_pair(row,column);

// check if pixel is noisy or dead (i.e. in the mask)
    const bool is_in = maskedPixels.find(pixel) != maskedPixels.end();
    if(!is_in){
//calibrate digi and store the new ones
      electrons = calibrate(detId,adc,row,column,pcalibrations);
      RPixCalibDigi calibDigi(row,column,adc,electrons);
      calib_rpix_digi_set_.insert(calibDigi);
    }
  }
  if(verbosity_) edm::LogInfo("RPixDetClusterizer")<<" RPix set size = "<<calib_rpix_digi_set_.size();

  SeedVector_.clear();

// clusterizing and storing the seeds
  for (const auto &rpcd : calib_rpix_digi_set_){
    if(rpcd.electrons() > SeedADCThreshold_*ElectronADCGain_){
      SeedVector_.push_back(rpcd);
    }
  }
  
  for(std::vector<RPixCalibDigi>::iterator SeedIt = SeedVector_.begin(); SeedIt!=SeedVector_.end();++SeedIt){
    make_cluster( *SeedIt, clusters);
  }
}

void RPixDetClusterizer::make_cluster(RPixCalibDigi aSeed,  std::vector<CTPPSPixelCluster> &clusters ){
/// check if seed already used
  if(calib_rpix_digi_set_.size()==0 || calib_rpix_digi_set_.find(aSeed)==calib_rpix_digi_set_.end() ){
    return;
  }
// creating a temporary cluster
  tempCluster atempCluster;

// filling the cluster with the seed
  atempCluster.addPixel(aSeed.row(),aSeed.column(),aSeed.electrons());
  calib_rpix_digi_set_.erase( aSeed );

  while ( ! atempCluster.empty()) {
  //This is the standard algorithm to find and add a pixel
    auto curInd = atempCluster.top(); 
    atempCluster.pop();
    for ( auto c = std::max(0,int(atempCluster.col[curInd])-1); c < std::min(int(atempCluster.col[curInd])+2,maxCol) ; ++c) {
      for ( auto r = std::max(0,int(atempCluster.row[curInd])-1); r < std::min(int(atempCluster.row[curInd])+2,maxRow); ++r)  {
	
	for(std::set<RPixCalibDigi>::iterator RPCDit2 = calib_rpix_digi_set_.begin(); RPCDit2 != calib_rpix_digi_set_.end(); ){
	  if( (*RPCDit2).column() == c && (*RPCDit2).row() == r && (*RPCDit2).electrons() > ADCThreshold_*ElectronADCGain_ ){

	    if(!atempCluster.addPixel( r,c,(*RPCDit2).electrons() )) {
	      CTPPSPixelCluster acluster(atempCluster.isize,atempCluster.adc, atempCluster.row,atempCluster.col, atempCluster.rowmin,atempCluster.colmin);
	      clusters.push_back(acluster);
	      return;
	    }
	    RPCDit2 =  calib_rpix_digi_set_.erase(RPCDit2);
	    
	  }else{
	    ++RPCDit2;
	  }
	}
      }
    }
	     
  }  // while accretion

  CTPPSPixelCluster cluster(atempCluster.isize,atempCluster.adc, atempCluster.row,atempCluster.col, atempCluster.rowmin,atempCluster.colmin);
  clusters.push_back(cluster);
}

int RPixDetClusterizer::calibrate(unsigned int detId, int adc, int row, int col, const CTPPSPixelGainCalibrations *pcalibrations){

  float gain=0;
  float pedestal=0;
  int electrons=0;
  
  if(!doSingleCalibration_){

    CTPPSPixelGainCalibration DetCalibs = pcalibrations->getGainCalibration(detId);

    if(DetCalibs.getDetId() != 0){
      gain = DetCalibs.getGain(col,row)*highRangeCal/lowRangeCal;
      pedestal = DetCalibs.getPed(col,row);
      float vcal = (adc - pedestal)*gain;
      electrons = int(vcal*VcaltoElectronGain_ + VcaltoElectronOffset_);
    }
    else{
      gain = ElectronADCGain_;
      pedestal = 0;
      electrons = int(adc*gain-pedestal);
    }


    if(verbosity_) edm::LogInfo("RPixCalibration") << "calibrate:  adc = " << adc << " electrons = " << electrons << " gain = " << gain << " pedestal = " << pedestal ;
  }else{
    gain = ElectronADCGain_;
    pedestal = 0;
    electrons = int(adc*gain-pedestal);
    if(verbosity_) edm::LogInfo("RPixCalibration") << "calibrate:  adc = " << adc << " electrons = " << electrons << " ElectronADCGain = " << gain << " pedestal = " << pedestal ;
  }
  if (electrons<0) {
    edm::LogInfo("RPixCalibration") << "RPixDetClusterizer::calibrate: *** electrons < 0 *** --> " << electrons << "  --> electrons = 0";
    electrons = 0;
  }

  return electrons;

}

