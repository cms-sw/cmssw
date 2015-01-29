#ifndef DTSegmentAnalysis_H
#define DTSegmentAnalysis_H

/** \class DTTriggerCheck
 *
 *  \author S.Bolognesi - INFN Torino
 */

#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/LTCDigi/interface/LTCDigi.h"

#include <string>
#include <map>
#include <vector>

class DQMStore;
class MonitorElement;

class DTTriggerCheck: public DQMEDAnalyzer{

friend class DTMonitorModule;
public:
  /// Constructor
  DTTriggerCheck(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTTriggerCheck();

/// Analyze
void analyze(const edm::Event& event, const edm::EventSetup& setup);

void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

protected:

private:

  bool debug;

  MonitorElement* histo;

  bool isLocalRun;
  edm::EDGetTokenT<LTCDigiCollection> ltcDigiCollectionToken_;
};
#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
