#ifndef JetMETDQMDCSFilter_H
#define JetMETDQMDCSFilter_H

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class JetMETDQMDCSFilter {
 public:
 JetMETDQMDCSFilter( const edm::ParameterSet & );
 ~JetMETDQMDCSFilter();
  bool filter(const edm::Event& , const edm::EventSetup& );
  private:

  bool verbose_;
  bool detectorOn_;
  std::string detectorTypes_;

};

#endif
