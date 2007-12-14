#ifndef EventFilter_SiStripRawToDigi_SiStripFEDRawDataAnalyzer_H
#define EventFilter_SiStripRawToDigi_SiStripFEDRawDataAnalyzer_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiStripFEDRawDataCheck;

/**
   @class SiStripFEDRawDataAnalyzer 
   @brief Analyzes contents of FEDRawData collection
*/
class SiStripFEDRawDataAnalyzer : public edm::EDAnalyzer {
  
 public:
  
  SiStripFEDRawDataAnalyzer( const edm::ParameterSet& );
  virtual ~SiStripFEDRawDataAnalyzer();

  void analyze( const edm::Event&, const edm::EventSetup& );

 private:

  void beginJob( edm::EventSetup const& ) {;}
  void endJob() {;}

  SiStripFEDRawDataCheck* check_;
  
};

#endif // EventFilter_SiStripRawToDigi_SiStripFEDRawDataAnalyzer_H

