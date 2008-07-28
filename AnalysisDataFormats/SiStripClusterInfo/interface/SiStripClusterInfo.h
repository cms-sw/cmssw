
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
#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
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
  
  // constructor with the detid (in case it will be eliminated from the SiStripCluster)
  SiStripClusterInfo(const uint32_t cluster_detId,
                     const SiStripCluster& cluster, 
                     const edm::EventSetup& es, 
		     std::string CMNSubtractionMode="Median");
  
  // constructor without the detid (taken from SiStripCluster)
  SiStripClusterInfo(const SiStripCluster& cluster, 
                     const edm::EventSetup& es, 
		     std::string CMNSubtractionMode="Median");
  
  ~SiStripClusterInfo();
    
  /** Cluster DetId */
  uint32_t getDetId() const {return cluster_detId_;};
  
  /** Cluster First Strip Number (the first strip is 0, this number can be used as index of a vector) */
  uint16_t                           getFirstStrip() const {return cluster_->firstStrip();}
  /** Cluster Position */
  float                                getPosition() const {return cluster_->barycenter();}
  /** Vector of Cluster Strip Charge */
  const std::vector<uint8_t>&  getStripAmplitudes() const {return cluster_->amplitudes();} 
  
  
  /** Cluster Width */
  float                                   getWidth() const {return cluster_->amplitudes().size();}
  /** Cluster Charge (Signal) */
  float                                  getCharge() const;
  /** Strip Number of the strip with maximum charge in the cluster (Cluster Seed)
      (the first strip is 0, this number can be used as index of a vector) */
  uint16_t                          getMaxPosition() const;
  /** Strip Number of the strip with maximum charge in the cluster (Cluster Seed)
      (the first strip is 0, this number can be used as index of a vector) */
  uint16_t                          getMaxPosition(const SiStripCluster*) const;
  /** Charge of the Strip with maximum charge in the cluster (Cluster Seed) */
  float                               getMaxCharge() const;  
  /** Sum of the cluster strip charges Left/Right the Cluster Seed */
  std::pair<float,float>               getChargeLR() const;
  /** Cluster Charge of the first strip Left/Right the Cluster Seed */
  std::pair<float,float> getChargeLRFirstNeighbour() const;
  //
  /** RawDigi Charge (Cluster Seed,First Left Strip,First Right Strip) */
  std::vector<float> getRawChargeCLR( const edm::DetSet<SiStripRawDigi>&,
				      edmNew::DetSet<SiStripCluster>&,
				      std::string );
  
  
  /** Vector of Cluster Strip Noise */
  std::vector<float>               getStripNoises() const; 
  /** Vector of Cluster Strip Noise rescaled by APV Gain */
  std::vector<float> getStripNoisesRescaledByGain() const; 
  /** Cluster Noise */
  float                                  getNoise()  ;
  /** Cluster Noise rescaled by APV Gain */
  float                    getNoiseRescaledByGain()  ;
  /** Strip Noise */
  float                        getNoiseForStripNb(uint16_t istrip ) const;
  
  /** Vector of APV Gain */
  std::vector<float> getApvGains() const; 
  /** Strip APV Gain */
  float        getGainForStripNb(uint16_t istrip ) const;
  
  /** Cluster Signal-to-Noise ratio (S/N) */
  float               getSignalOverNoise() ;
  /** Cluster Signal-to-Noise ratio (S/N) rescaled by APV Gain */
  float getSignalOverNoiseRescaledByGain() ;

  
  /** Vectors of the cluster strip charges Left/Right the Cluster Seed from RawDigis */
  std::pair< std::vector<float>,std::vector<float> > getRawDigiAmplitudesLR(      uint32_t&                          neighbourStripNr, 
                                                                            const edm::DetSetVector<SiStripRawDigi>&    rawDigis_dsv_, 
		                                                                  edmNew::DetSetVector<SiStripCluster>& clusters_dsv_,
		                                                                  std::string                           rawDigiLabel);

  /** Vectors of the cluster strip charges Left/Right the Cluster Seed from RawDigis */
  std::pair< std::vector<float>,std::vector<float> > getRawDigiAmplitudesLR(      uint32_t&                   neighbourStripNr, 
                                                                            const edm::DetSet<SiStripRawDigi>&    rawDigis_ds_, 
                                                                                  edmNew::DetSet<SiStripCluster>& clusters_ds_,
                                                                                  std::string                     rawDigiLabel); 


  
  /** Vectors of the cluster strip charges Left/Right the Cluster Seed from Digis */
  std::pair< std::vector<float>,std::vector<float> >  getDigiAmplitudesLR(      uint32_t&                          neighbourStripNr,
                                                                          const edm::DetSetVector<SiStripDigi>&          digis_dsv_,
										edmNew::DetSetVector<SiStripCluster>& clusters_dsv_);

  /** Vectors of the cluster strip charges Left/Right the Cluster Seed from Digis */
  std::pair< std::vector<float>,std::vector<float> > getDigiAmplitudesLR(      uint32_t&                   neighbourStripNr, 
                                                                         const edm::DetSet<SiStripDigi>&          digis_ds_, 
                                                                               edmNew::DetSet<SiStripCluster>& clusters_ds_);
 

 private:
  
  
  void rawdigi_algorithm(const edm::DetSet<SiStripRawDigi>    rawDigis_ds_,
			       edmNew::DetSet<SiStripCluster> clusters_ds_,
			       std::string                    rawDigiLabel);
  
  void digi_algorithm(const edm::DetSet<SiStripDigi>         digis_ds_,
                            edmNew::DetSet<SiStripCluster> cluster_ds_);
  
  void findNeigh(char*                                  mode,
                 edmNew::DetSet<SiStripCluster> clusters_ds_,
		 std::vector<int16_t>&                  vadc,
		 std::vector<int16_t>&                vstrip);
 
  const edm::EventSetup&      es_; 
  edm::ESHandle<SiStripNoises>        noiseHandle_;
  edm::ESHandle<SiStripGain>           gainHandle_;
  edm::ESHandle<SiStripPedestals> pedestalsHandle_;
    
  const SiStripCluster *cluster_;
  uint32_t cluster_detId_;
   
  uint32_t neighbourStripNr_;
  SiStripCommonModeNoiseSubtractor* SiStripCommonModeNoiseSubtractor_;
  std::string CMNSubtractionMode_;
  bool validCMNSubtraction_;  
  SiStripPedestalsSubtractor* SiStripPedestalsSubtractor_;
  float                amplitudeC_;
  std::vector<float>   amplitudesL_;
  std::vector<float>   amplitudesR_;
  
};

#endif // ANALYSISDATAFORMATS_SISTRIPCLUSTER_H
