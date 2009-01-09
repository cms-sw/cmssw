#ifndef ScaleDBReader_H
#define ScaleDBReader_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MuonAnalysis/MomentumScaleCalibration/interface/MomentumScaleCorrector.h"

class ScaleDBReader : public edm::EDAnalyzer {

 public:
  explicit ScaleDBReader( const edm::ParameterSet& );
  ~ScaleDBReader();

  void beginJob ( const edm::EventSetup& iSetup );
  
  void analyze( const edm::Event&, const edm::EventSetup& );

 private:
  //  uint32_t printdebug_;
  auto_ptr<MomentumScaleCorrector> corrector_;

};
#endif
