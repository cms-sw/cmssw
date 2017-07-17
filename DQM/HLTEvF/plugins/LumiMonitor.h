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

class LumiMonitor : public DQMEDAnalyzer 
{
public:
  LumiMonitor( const edm::ParameterSet& );
  ~LumiMonitor() = default;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);

protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:

  static MEbinning getHistoPSet  (edm::ParameterSet pset);
  static MEbinning getHistoLSPSet(edm::ParameterSet pset);

  std::string folderName_;

  edm::EDGetTokenT<LumiScalersCollection> lumiScalersToken_;
  MEbinning lumi_binning_;
  MEbinning ls_binning_;

  bool  doPixelLumi_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClustersToken_;
  bool  useBPixLayer1_;
  int   minNumberOfPixelsPerCluster_;
  float minPixelClusterCharge_;	
  MEbinning pixelCluster_binning_;
  MEbinning pixellumi_binning_;


  edm::EDGetTokenT<LumiSummary> lumiSummaryToken_;
  
  MonitorElement* numberOfPixelClustersVsLS_;
  MonitorElement* numberOfPixelClustersVsLumi_;
  MonitorElement* lumiVsLS_;
  MonitorElement* pixelLumiVsLS_;
  MonitorElement* pixelLumiVsLumi_;

  float lumi_factor_per_bx_;

};

#endif // LUMIMONITOR_H
