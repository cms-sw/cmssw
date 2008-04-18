
#ifndef ANALYSISDATAFORMATS_SISTRIPCLUSTERINFO_H
#define ANALYSISDATAFORMATS_SISTRIPCLUSTERINFO_H
// -*- C++ -*-
//
// Package:     AnalysisDataFormats
// Class  :     SiStripClusterInfo
// 
/**\class SiStripClusterInfo SiStripClusterInfo.h AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h

 Description: utility class gathering all access methods to SiStripCluster-related information
              for detector-related studies and DQM

*/
//
// Original Author:  Evelyne Delmeire
//         Created:  Sun Jan 27 16:07:51 CET 2007
//
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"

#include "CommonTools/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"

#include <vector>
#include <sstream>
#include <string>
#include "boost/cstdint.hpp"

class SiStripRawDigi;

class SiStripDigi;

class SiStripClusterInfo {
  
 public:
  
  SiStripClusterInfo(const uint32_t cluster_detId,
                     const SiStripCluster& cluster, 
                     const edm::EventSetup& es, 
		     std::string CMNSubtractionMode="Median");
  
  ~SiStripClusterInfo();
    
  uint16_t                           getFirstStrip() const {return cluster_->firstStrip();}
  float                                getPosition() const {return cluster_->barycenter();}
  const std::vector<uint8_t>&  getStripAmplitudes() const {return cluster_->amplitudes();} 
  
    
  float                            getWidth() const {return cluster_->amplitudes().size();}
  float                           getCharge() ;
  uint16_t                   getMaxPosition() ;
  float                        getMaxCharge() ;  
  std::pair<float,float>        getChargeLR() ;
  
  
  std::vector<float>               getStripNoises() const; 
  std::vector<float> getStripNoisesRescaledByGain() const; 
  float                                  getNoise()  ;
  float                    getNoiseRescaledByGain()  ;
  float                        getNoiseForStripNb(uint16_t istrip ) const;
  
  std::vector<float> getApvGains() const; 
  float        getGainForStripNb(uint16_t istrip ) const;
  
  float               getSignalOverNoise() ;
  float getSignalOverNoiseRescaledByGain() ;

  
  std::pair< std::vector<float>,std::vector<float> > getRawDigiAmplitudesLR(      uint32_t&                       neighbourStripNr, 
                                                                            const edm::DetSetVector<SiStripRawDigi>& rawDigis_dsv_, 
		                                                                  edm::DetSetVector<SiStripCluster>& clusters_dsv_,
		                                                                  std::string                         rawDigiLabel);

  std::pair< std::vector<float>,std::vector<float> > getRawDigiAmplitudesLR(      uint32_t&                neighbourStripNr, 
                                                                            const edm::DetSet<SiStripRawDigi>& rawDigis_ds_, 
                                                                                  edm::DetSet<SiStripCluster>& clusters_ds_,
                                                                                  std::string                  rawDigiLabel); 


  
  std::pair< std::vector<float>,std::vector<float> >  getDigiAmplitudesLR(      uint32_t&                       neighbourStripNr,
                                                                          const edm::DetSetVector<SiStripDigi>&       digis_dsv_,
									       edm::DetSetVector<SiStripCluster>&  clusters_dsv_);

  std::pair< std::vector<float>,std::vector<float> > getDigiAmplitudesLR(      uint32_t&                neighbourStripNr, 
                                                                         const edm::DetSet<SiStripDigi>&       digis_ds_, 
                                                                               edm::DetSet<SiStripCluster>& clusters_ds_);
 

 private:
  
  
  void rawdigi_algorithm(const edm::DetSet<SiStripRawDigi> rawDigis_ds_,
			       edm::DetSet<SiStripCluster> clusters_ds_,
			       std::string                 rawDigiLabel);
  
  void digi_algorithm(const edm::DetSet<SiStripDigi>     digis_ds_,
                           edm::DetSet<SiStripCluster> cluster_ds_);
  
  void findNeigh(char*                               mode,
                 edm::DetSet<SiStripCluster> clusters_ds_,
		 std::vector<int16_t>&               vadc,
		 std::vector<int16_t>&             vstrip);
 
  const edm::EventSetup&      es_; 
  edm::ESHandle<SiStripNoises>        noiseHandle_;
  edm::ESHandle<SiStripGain>           gainHandle_;
  edm::ESHandle<SiStripPedestals> pedestalsHandle_;
    
  const SiStripCluster *cluster_;
  uint32_t cluster_detId_;
   
  uint16_t neighbourStripNr;
  SiStripCommonModeNoiseSubtractor* SiStripCommonModeNoiseSubtractor_;
  std::string CMNSubtractionMode_;
  bool validCMNSubtraction_;  
  SiStripPedestalsSubtractor* SiStripPedestalsSubtractor_;
  std::vector<float>   amplitudesL_;
  std::vector<float>   amplitudesR_;
 
  
};

#endif // ANALYSISDATAFORMATS_SISTRIPCLUSTER_H
