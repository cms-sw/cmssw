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
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"


//
// class declaration
//

class LumiMonitor : public DQMEDAnalyzer 
{
public:
  LumiMonitor( const edm::ParameterSet& );
  ~LumiMonitor();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);

protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup);

private:

  void bookHistograms();

  edm::ParameterSet conf_;

  std::string folderName_;

  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClustersToken_;
  edm::EDGetTokenT<LumiScalersCollection> lumiscalersToken_;
  edm::EDGetTokenT<LumiSummary> lumiSummaryToken_;
  
  MonitorElement* numberOfPixelClustersVsLS_;
  MonitorElement* numberOfPixelClustersVsLumi_;
  MonitorElement* lumiVsLS_;
  MonitorElement* pixelLumiVsLS_;
  MonitorElement* pixelLumiVsLumi_;

  bool  doPixelLumi_;
  bool  useBPixLayer1_;
  int   minNumberOfPixelsPerCluster_;
  float minPixelClusterCharge_;	

  float lumi_factor_per_bx_;

  unsigned long long m_cacheID_;
};

#endif // LUMIMONITOR_H
