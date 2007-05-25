#ifndef DQM_SiStripCommon_FEDRawDataAnalyzer_H
#define DQM_SiStripCommon_FEDRawDataAnalyzer_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "boost/cstdint.hpp"
#include <vector>
#include <string>

/**
   @class FEDRawDataAnalyzer 
   @brief Analyzes contents of FEDRawData collection
*/
class FEDRawDataAnalyzer : public edm::EDAnalyzer {
  
 public:
  
  FEDRawDataAnalyzer( const edm::ParameterSet& );
  virtual ~FEDRawDataAnalyzer() {;}

  void beginJob( edm::EventSetup const& );
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob();
  
 private:

  class Temp {
  public:
    Temp( float rate, float meas, int nfeds ) : 
      rate_(rate), meas_(meas), nfeds_(nfeds) {;}
    Temp() : rate_(0.), meas_(0.), nfeds_(0) {;}
    float rate_;
    float meas_;
    int nfeds_;
  };

  std::vector<Temp> temp_;
  
  std::string label_;
  std::string instance_;
  int32_t sleep_;
  
};

#endif // DQM_SiStripCommon_FEDRawDataAnalyzer_H

