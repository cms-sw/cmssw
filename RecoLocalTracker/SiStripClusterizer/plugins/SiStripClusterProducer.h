#ifndef RecoLocalTracker_SiStripClusterizer_SiStripClusterProducer_h
#define RecoLocalTracker_SiStripClusterizer_SiStripClusterProducer_h

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerFactory.h"

class SiStripClusterProducer : public edm::EDProducer {
  
 public:
    
  typedef SiStripClusterizerFactory::DigisDSV DigisDSV;
  typedef SiStripClusterizerFactory::ClustersDSV ClustersDSV;
  typedef SiStripClusterizerFactory::ClustersDSVnew ClustersDSVnew;
  
  explicit SiStripClusterProducer( const edm::ParameterSet& );
    
  ~SiStripClusterProducer();

  void produce( edm::Event&, const edm::EventSetup& );
  void beginRun( edm::Run&, const edm::EventSetup& );
  
 private:
    
  SiStripClusterizerFactory* clusterizer_;

  edm::InputTag tag_;

  bool edmNew_;

};

#endif // RecoLocalTracker_SiStripClusterizer_SiStripClusterProducer_h
