#ifndef DQM_BeamMonitor_OnlineBeamMonitor_h
#define DQM_BeamMonitor_OnlineBeamMonitor_h

/** \class OnlineBeamMonitor
 * *
 *  \author   Simone Gennai INFN/Bicocca
 */
// C++
#include <map>
#include <vector>
#include <string>
#include <fstream>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineLegacyObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineHLTObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotTransientObjectsRcd.h"

namespace onlinebeammonitor {
  struct BeamSpotInfo {
    typedef std::map<std::string, reco::BeamSpot> BeamSpotContainer;
    BeamSpotContainer beamSpotsMap_;
  };
}  // namespace onlinebeammonitor

class OnlineBeamMonitor : public DQMOneEDAnalyzer<edm::LuminosityBlockCache<onlinebeammonitor::BeamSpotInfo>> {
public:
  OnlineBeamMonitor(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  std::shared_ptr<onlinebeammonitor::BeamSpotInfo> globalBeginLuminosityBlock(
      const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup) const override;
  void globalEndLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup) override;
  void dqmEndRun(edm::Run const&, edm::EventSetup const&) override;

private:
  //Typedefs
  //                BF,BS...
  typedef std::map<std::string, reco::BeamSpot> BeamSpotContainer;
  //                x,y,z,sigmax(y,z)... [run,lumi]          Histo name
  typedef std::map<std::string, std::map<std::string, std::map<std::string, MonitorElement*>>> HistosContainer;
  //                x,y,z,sigmax(y,z)... [run,lumi]          Histo name
  typedef std::map<std::string, std::map<std::string, std::map<std::string, int>>> PositionContainer;

  //Parameters
  std::string monitorName_;
  edm::ESGetToken<BeamSpotObjects, BeamSpotTransientObjectsRcd> bsTransientToken_;
  edm::ESGetToken<BeamSpotOnlineObjects, BeamSpotOnlineHLTObjectsRcd> bsHLTToken_;
  edm::ESGetToken<BeamSpotOnlineObjects, BeamSpotOnlineLegacyObjectsRcd> bsLegacyToken_;
  std::ofstream fasciiDIP;

  //Service variables
  int numberOfValuesToSave_;
  std::vector<int> processedLumis_;
  // MonitorElements:
  MonitorElement* bsChoice_;

  //Containers
  HistosContainer histosMap_;
  PositionContainer positionsMap_;
  std::vector<std::string> varNamesV_;                            //x,y,z,sigmax(y,z)
  std::multimap<std::string, std::string> histoByCategoryNames_;  //run, lumi

  //For File Writing
  bool appendRunTxt_;
  bool writeDIPTxt_;
  std::string outputDIPTxt_;
};

#endif
