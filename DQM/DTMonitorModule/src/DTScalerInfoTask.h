#ifndef DTScalerInfoTask_H
#define DTScalerInfoTask_H

/*
 * \file DTScalerInfoTask.h
 *
 * \author C. Battilana - CIEMAT
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "DataFormats/DTDigi/interface/DTLocalTriggerCollection.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"

#include <vector>
#include <string>
#include <map>

class DTTimeEvolutionHisto;

class DTScalerInfoTask: public DQMEDAnalyzer{

  friend class DTMonitorModule;

 public:

  /// Constructor
  DTScalerInfoTask(const edm::ParameterSet& ps );

  /// Destructor
  virtual ~DTScalerInfoTask();

 protected:

  // Book the histograms
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  ///Beginrun
  void dqmBeginRun(const edm::Run& , const edm::EventSetup&);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// To reset the MEs
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) ;

  /// Perform trend plot operations
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) ;

 private:

  int nEvents;
  int nEventsInLS;

  edm::ParameterSet theParams;

  edm::EDGetTokenT<LumiScalersCollection> scalerToken_;

  std::map<std::string ,DTTimeEvolutionHisto* > trendHistos;
  MonitorElement* nEventMonitor;

};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
