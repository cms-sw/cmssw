#ifndef DBReader_H
#define DBReader_H

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
#include "MuonAnalysis/MomentumScaleCalibration/interface/ResolutionFunction.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/BackgroundFunction.h"

class DBReader : public edm::EDAnalyzer {

 public:
  explicit DBReader( const edm::ParameterSet& );
  ~DBReader();

  void beginJob ( const edm::EventSetup& iSetup );
  
  void analyze( const edm::Event&, const edm::EventSetup& );

 private:
  //  uint32_t printdebug_;
  string type_;
  //auto_ptr<BaseFunction> corrector_;
  auto_ptr<MomentumScaleCorrector> corrector_;
  auto_ptr<ResolutionFunction> resolution_;
  auto_ptr<BackgroundFunction> background_;

};
#endif
