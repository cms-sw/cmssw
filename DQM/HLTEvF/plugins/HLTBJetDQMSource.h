#ifndef DQM_HLTEvF_HLTBJetDQMSource_H
#define DQM_HLTEvF_HLTBJetDQMSource_H

/** \class HLTBJetDQMSource
 * *
 *  DQM source for BJet HLT paths
 *
 *  $Date: 2008/05/13 16:38:49 $
 *  $Revision: 1.3 $
 *  \author Andrea Bocci, Pisa
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DQMStore;
class MonitorElement;
class edm::Event;
class edm::EventSetup;
class edm::LuminosityBlock;
class edm::Run;

//
// class declaration
//

class HLTBJetDQMSource : public edm::EDAnalyzer {
public:
  HLTBJetDQMSource(const edm::ParameterSet & config);
  ~HLTBJetDQMSource();

protected:

  void beginJob(const edm::EventSetup & setup);
  void endJob();

  void beginRun(const edm::Run & run, const edm::EventSetup & setup);
  void endRun(const edm::Run & run, const edm::EventSetup & setup);

  void beginLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup);
  void endLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup);

  void analyze(const edm::Event & event, const edm::EventSetup & setup);
  void analyzeLifetime(const edm::Event & event, const edm::EventSetup & setup);
  void analyzeSoftmuon(const edm::Event & event, const edm::EventSetup & setup);
  void analyzePerformance(const edm::Event & event, const edm::EventSetup & setup);

private:

  MonitorElement * book(const char * name, const char * title , int x_bins, double x_min, double x_max, const char * x_axis = 0);
  MonitorElement * book(const char * name, const char * title , int x_bins, double x_min, double x_max, int y_bins, double y_min, double y_max, const char * x_axis = 0, const char * y_axis = 0);

  edm::InputTag m_lifetimeL2Jets;
  edm::InputTag m_lifetimeL25TagInfo;
  edm::InputTag m_lifetimeL25JetTags;
  edm::InputTag m_lifetimeL3TagInfo;
  edm::InputTag m_lifetimeL3JetTags;
  edm::InputTag m_softmuonL2Jets;
  edm::InputTag m_softmuonL25TagInfo;
  edm::InputTag m_softmuonL25JetTags;
  edm::InputTag m_softmuonL3TagInfo;
  edm::InputTag m_softmuonL3JetTags;
  edm::InputTag m_performanceL2Jets;
  edm::InputTag m_performanceL25TagInfo;
  edm::InputTag m_performanceL25JetTags;
  edm::InputTag m_performanceL3TagInfo;
  edm::InputTag m_performanceL3JetTags;

  std::string m_monitorName;
  std::string m_outputFile;
  bool m_storeROOT;
  edm::Service<DQMStore> m_dbe;

  MonitorElement * m_lifetimeL2JetsEnergy;
  MonitorElement * m_lifetimeL2JetsET;
  MonitorElement * m_lifetimeL2JetsEta;
  MonitorElement * m_lifetimeL2JetsPhi;
  MonitorElement * m_lifetimeL25JetsEnergy;
  MonitorElement * m_lifetimeL25JetsET;
  MonitorElement * m_lifetimeL25JetsEta;
  MonitorElement * m_lifetimeL25JetsPhi;
  MonitorElement * m_lifetimeL25TrackMultiplicity;
  MonitorElement * m_lifetimeL25TrackHits;
  MonitorElement * m_lifetimeL25TrackChi2;
  MonitorElement * m_lifetimeL25IP2ndTrack2d;
  MonitorElement * m_lifetimeL25IP2ndTrack2dSig;
  MonitorElement * m_lifetimeL25IP2ndTrack3d;
  MonitorElement * m_lifetimeL25IP2ndTrack3dSig;
  MonitorElement * m_lifetimeL25IP3ndTrack2d;
  MonitorElement * m_lifetimeL25IP3ndTrack2dSig;
  MonitorElement * m_lifetimeL25IP3ndTrack3d;
  MonitorElement * m_lifetimeL25IP3ndTrack3dSig;
  MonitorElement * m_lifetimeL25Discriminator;
  MonitorElement * m_lifetimeL3JetsEnergy;
  MonitorElement * m_lifetimeL3JetsET;
  MonitorElement * m_lifetimeL3JetsEta;
  MonitorElement * m_lifetimeL3JetsPhi;
  MonitorElement * m_lifetimeL3TrackMultiplicity;
  MonitorElement * m_lifetimeL3TrackHits;
  MonitorElement * m_lifetimeL3TrackChi2;
  MonitorElement * m_lifetimeL3IP2ndTrack2d;
  MonitorElement * m_lifetimeL3IP2ndTrack2dSig;
  MonitorElement * m_lifetimeL3IP2ndTrack3d;
  MonitorElement * m_lifetimeL3IP2ndTrack3dSig;
  MonitorElement * m_lifetimeL3IP3ndTrack2d;
  MonitorElement * m_lifetimeL3IP3ndTrack2dSig;
  MonitorElement * m_lifetimeL3IP3ndTrack3d;
  MonitorElement * m_lifetimeL3IP3ndTrack3dSig;
  MonitorElement * m_lifetimeL3Discriminator;

  MonitorElement * m_softmuonL2JetsEnergy;
  MonitorElement * m_softmuonL2JetsET;
  MonitorElement * m_softmuonL2JetsEta;
  MonitorElement * m_softmuonL2JetsPhi;
  MonitorElement * m_softmuonL25JetsEnergy;
  MonitorElement * m_softmuonL25JetsET;
  MonitorElement * m_softmuonL25JetsEta;
  MonitorElement * m_softmuonL25JetsPhi;
  MonitorElement * m_softmuonL25MuonMultiplicity;
  MonitorElement * m_softmuonL25MuonHits;
  MonitorElement * m_softmuonL25MuonChi2;
  MonitorElement * m_softmuonL25MuonDeltaR;
  MonitorElement * m_softmuonL25MuonIP2dSig;
  MonitorElement * m_softmuonL25MuonIP3dSig;
  MonitorElement * m_softmuonL25MuonPtrel;
  MonitorElement * m_softmuonL25Discriminator;
  MonitorElement * m_softmuonL3JetsEnergy;
  MonitorElement * m_softmuonL3JetsET;
  MonitorElement * m_softmuonL3JetsEta;
  MonitorElement * m_softmuonL3JetsPhi;
  MonitorElement * m_softmuonL3MuonMultiplicity;
  MonitorElement * m_softmuonL3MuonHits;
  MonitorElement * m_softmuonL3MuonChi2;
  MonitorElement * m_softmuonL3MuonDeltaR;
  MonitorElement * m_softmuonL3MuonIP2dSig;
  MonitorElement * m_softmuonL3MuonIP3dSig;
  MonitorElement * m_softmuonL3MuonPtrel;
  MonitorElement * m_softmuonL3Discriminator;

  MonitorElement * m_performanceL2JetsEnergy;
  MonitorElement * m_performanceL2JetsET;
  MonitorElement * m_performanceL2JetsEta;
  MonitorElement * m_performanceL2JetsPhi;
  MonitorElement * m_performanceL25JetsEnergy;
  MonitorElement * m_performanceL25JetsET;
  MonitorElement * m_performanceL25JetsEta;
  MonitorElement * m_performanceL25JetsPhi;
  MonitorElement * m_performanceL25MuonMultiplicity;
  MonitorElement * m_performanceL25MuonHits;
  MonitorElement * m_performanceL25MuonChi2;
  MonitorElement * m_performanceL25MuonDeltaR;
  MonitorElement * m_performanceL25MuonIP2dSig;
  MonitorElement * m_performanceL25MuonIP3dSig;
  MonitorElement * m_performanceL25MuonPtrel;
  MonitorElement * m_performanceL25Discriminator;
  MonitorElement * m_performanceL3JetsEnergy;
  MonitorElement * m_performanceL3JetsET;
  MonitorElement * m_performanceL3JetsEta;
  MonitorElement * m_performanceL3JetsPhi;
  MonitorElement * m_performanceL3MuonMultiplicity;
  MonitorElement * m_performanceL3MuonHits;
  MonitorElement * m_performanceL3MuonChi2;
  MonitorElement * m_performanceL3MuonDeltaR;
  MonitorElement * m_performanceL3MuonIP2dSig;
  MonitorElement * m_performanceL3MuonIP3dSig;
  MonitorElement * m_performanceL3MuonPtrel;
  MonitorElement * m_performanceL3Discriminator;

  unsigned int m_counterEvent;
  unsigned int m_prescaleEvent;
};

#endif // DQM_HLTEvF_HLTBJetDQMSource_H
