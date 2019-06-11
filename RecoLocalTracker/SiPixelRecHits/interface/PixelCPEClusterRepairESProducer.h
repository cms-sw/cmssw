#ifndef RecoLocaltracker_SiPixelRecHits_PixelCPEClusterRepairESProducer_h
#define RecoLocaltracker_SiPixelRecHits_PixelCPEClusterRepairESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include <memory>

class PixelCPEClusterRepairESProducer : public edm::ESProducer {
public:
  PixelCPEClusterRepairESProducer(const edm::ParameterSet &p);
  ~PixelCPEClusterRepairESProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  std::unique_ptr<PixelClusterParameterEstimator> produce(const TkPixelCPERecord &);

private:
  edm::ParameterSet pset_;
  bool DoLorentz_;
};

#endif
