#ifndef DetectorStateFilter_H
#define DetectorStateFilter_H

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"

 
class DetectorStateFilter : public edm::EDFilter {
 public:
 DetectorStateFilter( const edm::ParameterSet & );
 ~DetectorStateFilter();
  private:
  bool filter( edm::Event &, edm::EventSetup const& );

  uint64_t nEvents_, nSelectedEvents_;
  bool verbose_;
  bool detectorOn_;
  std::string detectorType_;
  edm::EDGetTokenT<DcsStatusCollection> dcsStatusLabel_;
};

#endif
