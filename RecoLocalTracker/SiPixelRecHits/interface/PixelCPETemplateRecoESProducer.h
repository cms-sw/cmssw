#ifndef RecoLocaltracker_SiPixelRecHits_PixelCPETemplateRecoESProducer_h
#define RecoLocaltracker_SiPixelRecHits_PixelCPETemplateRecoESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include <memory>

class  PixelCPETemplateRecoESProducer: public edm::ESProducer{
 public:
  PixelCPETemplateRecoESProducer(const edm::ParameterSet & p);
  ~PixelCPETemplateRecoESProducer() override; 
  std::shared_ptr<PixelClusterParameterEstimator> produce(const TkPixelCPERecord &);
 private:
  std::shared_ptr<PixelClusterParameterEstimator> cpe_;
  edm::ParameterSet pset_;
  bool DoLorentz_;
};


#endif




