#ifndef RecoLocalTracker_SiStripClusterizer_SiStripClusterizerFactory_H
#define RecoLocalTracker_SiStripClusterizer_SiStripClusterizerFactory_H

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerAlgo.h"

class SiStripDetCabling;
class SiStripZeroSuppressorFactory; 

/** 
    @author M.Wingham, R.Bainbridge
    @class SiStripClusterizerFactory
    @brief Factory for clusterization algorithms. 
*/

class SiStripClusterizerFactory {

 public:

  typedef std::list<std::string> RegisteredAlgorithms; 
  
  typedef SiStripClusterizerAlgo::DigisDSV DigisDSV;
  typedef SiStripClusterizerAlgo::DigisDSVnew DigisDSVnew;
  
  typedef edm::DetSetVector<SiStripRawDigi> RawDigisDSV;
  typedef edmNew::DetSetVector<SiStripRawDigi> RawDigisDSVnew;

  typedef SiStripClusterizerAlgo::ClustersDSV ClustersDSV;
  typedef SiStripClusterizerAlgo::ClustersDSVnew ClustersDSVnew;
  
  SiStripClusterizerFactory( const edm::ParameterSet& );
  
  ~SiStripClusterizerFactory();

  // ----- DigisToClusters -----

  /// Digis (new DSV) to Clusters (new DSV)
  void clusterize( const DigisDSVnew&, ClustersDSVnew& );

  /// Digis (old DSV) to Clusters (new DSV)
  void clusterize( const DigisDSV&, ClustersDSVnew& );
  
  /// Digis (old DSV) to Clusters (old DSV)
  void clusterize( const DigisDSV&, ClustersDSV& );

  // ----- RawDigisToClusters -----
  
  /// RawDigis (new DSV) to Clusters (new DSV)
  void clusterize( const RawDigisDSVnew&, ClustersDSVnew& );

  /// RawDigis (old DSV) to Clusters (new DSV)
  void clusterize( const RawDigisDSV&, ClustersDSVnew& );

  /// RawDigis (old DSV) to Clusters (old DSV)
  void clusterize( const RawDigisDSV&, ClustersDSV& );
  
  /// Provides access to calibration constants for algorithm 
  void eventSetup( const edm::EventSetup& );
  
  /// Access to the algorithm object
  inline SiStripClusterizerAlgo* const algorithm() const;
  
 private:

  RegisteredAlgorithms algorithms_;

  SiStripClusterizerAlgo* algorithm_;
  
  SiStripZeroSuppressorFactory* factory_;

};

SiStripClusterizerAlgo* const SiStripClusterizerFactory::algorithm() const { return algorithm_; }

#endif // RecoLocalTracker_SiStripClusterizer_SiStripClusterizerFactory_H
