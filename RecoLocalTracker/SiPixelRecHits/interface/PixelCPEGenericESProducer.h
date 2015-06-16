#ifndef RecoLocaltracker_SiPixelRecHits_PixelCPEGenericESProducer_h
#define RecoLocaltracker_SiPixelRecHits_PixelCPEGenericESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include <boost/shared_ptr.hpp>

class  PixelCPEGenericESProducer: public edm::ESProducer{
 public:
  PixelCPEGenericESProducer(const edm::ParameterSet & p);
  virtual ~PixelCPEGenericESProducer(); 
  boost::shared_ptr<PixelClusterParameterEstimator> produce(const TkPixelCPERecord &);
 private:
  boost::shared_ptr<PixelClusterParameterEstimator> cpe_;
  edm::ParameterSet pset_;
  edm::ESInputTag magname_;
  bool useLAWidthFromDB_;
  bool useLAAlignmentOffsets_;
  bool UseErrorsFromTemplates_;
};


#endif




