#ifndef LUMIMONITOR_H
#define LUMIMONITOR_H

#include <string>
#include <vector>
#include <map>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"

//DataFormats

// legacy/stage-1 L1T:
//#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

// stage-2 L1T:
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"

// PS service
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/PrescaleService/interface/PrescaleService.h"

struct MEbinning {
  int nbins;
  double xmin;
  double xmax;
  //  MEbinning() {};
  //  explicit MEbinning(int n, double min, double max) { nbins= n; xmin = min; xmax = max;}
};

//
// class declaration
//

class PSMonitor : public DQMEDAnalyzer 
{
public:
  PSMonitor( const edm::ParameterSet& );
  ~PSMonitor() = default;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset, int value);

protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:

  void getHistoPSet(edm::ParameterSet& pset, MEbinning& mebinning);

  std::string folderName_;

  edm::EDGetTokenT<GlobalAlgBlkBxCollection> ugtBXToken_;
  
  /// Prescale service
  edm::service::PrescaleService* psService_;

  MonitorElement* psColumnIndexVsLS_;

  MEbinning ps_binning_;
  MEbinning ls_binning_;

};

#endif // LUMIMONITOR_H
