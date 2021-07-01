#ifndef DQM_BeamMonitor_AlcaBeamMonitor_h
#define DQM_BeamMonitor_AlcaBeamMonitor_h

/** \class AlcaBeamMonitor
 * *
 *  \author  Lorenzo Uplegger/FNAL
 *   modified by Simone Gennai INFN/Bicocca
 */
// C++
#include <map>
#include <vector>
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

class BeamFitter;
class PVFitter;

namespace alcabeammonitor {
  struct NoCache {};
}  // namespace alcabeammonitor

class AlcaBeamMonitor : public DQMOneEDAnalyzer<edm::LuminosityBlockCache<alcabeammonitor::NoCache>> {
public:
  AlcaBeamMonitor(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  std::shared_ptr<alcabeammonitor::NoCache> globalBeginLuminosityBlock(const edm::LuminosityBlock& iLumi,
                                                                       const edm::EventSetup& iSetup) const override;
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
  const edm::EDGetTokenT<reco::VertexCollection> primaryVertexLabel_;
  const edm::EDGetTokenT<reco::TrackCollection> trackLabel_;
  const edm::EDGetTokenT<reco::BeamSpot> scalerLabel_;
  const edm::ESGetToken<BeamSpotObjects, BeamSpotObjectsRcd> beamSpotToken_;
  bool perLSsaving_;  //to avoid nanoDQMIO crashing, driven by  DQMServices/Core/python/DQMStore_cfi.py

  //Service variables
  int numberOfValuesToSave_;
  std::unique_ptr<BeamFitter> theBeamFitter_;
  std::unique_ptr<PVFitter> thePVFitter_;
  mutable int numberOfProcessedLumis_;
  mutable std::vector<int> processedLumis_;

  // MonitorElements:
  MonitorElement* hD0Phi0_;
  MonitorElement* hDxyBS_;
  //mutable MonitorElement* theValuesContainer_;

  //Containers
  mutable BeamSpotContainer beamSpotsMap_;
  HistosContainer histosMap_;
  PositionContainer positionsMap_;
  std::vector<std::string> varNamesV_;                            //x,y,z,sigmax(y,z)
  std::multimap<std::string, std::string> histoByCategoryNames_;  //run, lumi
  mutable std::vector<reco::VertexCollection> vertices_;
};

#endif
