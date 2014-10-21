#ifndef DetStatus_H
#define DetStatus_H

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
 
class DetStatus : public edm::EDFilter {
 public:
 DetStatus( const edm::ParameterSet & );
 ~DetStatus();
  private:
  bool filter(edm::Event&, edm::EventSetup const&) override;

  bool verbose_;
  bool applyfilter_;
  bool AndOr_;
  std::vector<std::string>  DetNames_;
  unsigned int DetMap_;
  edm::EDGetTokenT<DcsStatusCollection> scalersToken_;
};

#endif
