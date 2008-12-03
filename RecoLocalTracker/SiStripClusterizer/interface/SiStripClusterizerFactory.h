#ifndef RecoLocalTracker_SiStripClusterizer_SiStripClusterizerFactory_H
#define RecoLocalTracker_SiStripClusterizer_SiStripClusterizerFactory_H

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerAlgo.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiStripDetCabling;
class SiStripZeroSuppressorFactory; 

/** 
    @author M.Wingham, D.Giordano, R.Bainbridge
    @class SiStripClusterizerFactory
    @brief Factory for clusterization algorithms. 
*/

class SiStripClusterizerFactory {

 public:

  typedef std::list<std::string> RegisteredAlgorithms; 
  
  SiStripClusterizerFactory( const edm::ParameterSet& );
  
  ~SiStripClusterizerFactory();
  
  /** Clusterization from zero-suppressed data for several DetIds. */
  void clusterize(const edm::DetSetVector<SiStripDigi>& digis, edm::DetSetVector<SiStripCluster>& clusters);

  /** Clusterization from zero-suppressed data for a single DetId. */
  void clusterize(const edm::DetSet<SiStripDigi>& digis, edm::DetSetVector<SiStripCluster>& clusters);
  
  /** Clusterization from raw data for several DetIds. */
  void clusterize(const edm::DetSetVector<SiStripRawDigi>& raw_digis, edm::DetSetVector<SiStripCluster>& clusters);
  
  /** Clusterization from raw data for a single DetId. */
  void clusterize(const edm::DetSet<SiStripRawDigi>& raw_digis, edm::DetSetVector<SiStripCluster>& clusters);

  /** Provides access to calibration constants for algorithm. */ 
  void eventSetup(const edm::EventSetup&);
  
  /** Access to the algorithm object. */
  inline SiStripClusterizerAlgo* const algorithm() const;
  
 private:

  RegisteredAlgorithms algorithms_;

  SiStripClusterizerAlgo* algorithm_;
  
  SiStripZeroSuppressorFactory* factory_;

};

SiStripClusterizerAlgo* const SiStripClusterizerFactory::algorithm() const { return algorithm_; }

#endif // RecoLocalTracker_SiStripClusterizer_SiStripClusterizerFactory_H
