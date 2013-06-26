#ifndef DQM_HLTEvF_HLTMonBTagIPSource_H
#define DQM_HLTEvF_HLTMonBTagIPSource_H

/** \class HLTMonBTagIPSource
 * *
 *  DQM source for BJet HLT paths
 *
 *  $Date: 2009/10/14 07:22:39 $
 *  $Revision: 1.7 $
 *  \author Andrea Bocci, Pisa
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DQMStore;
class MonitorElement;

//
// class declaration
//

class HLTMonBTagIPSource : public edm::EDAnalyzer {
public:
  HLTMonBTagIPSource(const edm::ParameterSet & config);
  ~HLTMonBTagIPSource();

protected:
  void beginJob(void);
  void endJob(void);

  void beginRun(const edm::Run & run, const edm::EventSetup & setup);
  void endRun(const edm::Run & run, const edm::EventSetup & setup);

  void beginLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup);
  void endLuminosityBlock(const edm::LuminosityBlock & lumi, const edm::EventSetup & setup);

  void analyze(const edm::Event & event, const edm::EventSetup & setup);

private:
  MonitorElement * book(const std::string & name, const std::string & title, int x_bins, double x_min, double x_max, const char * x_axis = 0);
  MonitorElement * book(const std::string & name, const std::string & title, int x_bins, double x_min, double x_max, int y_bins, double y_min, double y_max, const char * x_axis = 0, const char * y_axis = 0);

  edm::InputTag m_L1Filter;
  edm::InputTag m_L2Filter;
  edm::InputTag m_L25Filter;
  edm::InputTag m_L3Filter;
  edm::InputTag m_L2Jets;
  edm::InputTag m_L25TagInfo;
  edm::InputTag m_L25JetTags;
  edm::InputTag m_L3TagInfo;
  edm::InputTag m_L3JetTags;

  edm::InputTag m_triggerResults;
  std::string m_processName;
  std::string m_pathName;
  std::string m_monitorName;
  std::string m_outputFile;
  bool m_storeROOT;
  unsigned int m_size;
  edm::Service<DQMStore> m_dbe;
  bool m_init;

  // tool to access the HLT confioguration
  unsigned int m_pathIndex;
  unsigned int m_L1FilterIndex;
  unsigned int m_L2FilterIndex;
  unsigned int m_L25FilterIndex;
  unsigned int m_L3FilterIndex;

  // MonitorElement's (plots) filled by the source
  MonitorElement * m_plotRates;

  MonitorElement * m_plotL2JetsEnergy;
  MonitorElement * m_plotL2JetsET;
  MonitorElement * m_plotL2JetsEta;
  MonitorElement * m_plotL2JetsPhi;
  MonitorElement * m_plotL2JetsEtaPhi;
  MonitorElement * m_plotL2JetsEtaET;
  MonitorElement * m_plotL25JetsEnergy;
  MonitorElement * m_plotL25JetsET;
  MonitorElement * m_plotL25JetsEta;
  MonitorElement * m_plotL25JetsPhi;
  MonitorElement * m_plotL25JetsEtaPhi;
  MonitorElement * m_plotL25JetsEtaET;
  MonitorElement * m_plotL25TrackMultiplicity;
  MonitorElement * m_plotL25TrackHits;
  MonitorElement * m_plotL25TrackChi2;
  MonitorElement * m_plotL25TrackEtaPhi;
  MonitorElement * m_plotL25TrackEtaPT;
  MonitorElement * m_plotL25IP2ndTrack2d;
  MonitorElement * m_plotL25IP2ndTrack2dSig;
  MonitorElement * m_plotL25IP2ndTrack3d;
  MonitorElement * m_plotL25IP2ndTrack3dSig;
  MonitorElement * m_plotL25IP3ndTrack2d;
  MonitorElement * m_plotL25IP3ndTrack2dSig;
  MonitorElement * m_plotL25IP3ndTrack3d;
  MonitorElement * m_plotL25IP3ndTrack3dSig;
  MonitorElement * m_plotL25Discriminator;
  MonitorElement * m_plotL3JetsEnergy;
  MonitorElement * m_plotL3JetsET;
  MonitorElement * m_plotL3JetsEta;
  MonitorElement * m_plotL3JetsPhi;
  MonitorElement * m_plotL3JetsEtaPhi;
  MonitorElement * m_plotL3JetsEtaET;
  MonitorElement * m_plotL3TrackMultiplicity;
  MonitorElement * m_plotL3TrackHits;
  MonitorElement * m_plotL3TrackChi2;
  MonitorElement * m_plotL3TrackEtaPhi;
  MonitorElement * m_plotL3TrackEtaPT;
  MonitorElement * m_plotL3IP2ndTrack2d;
  MonitorElement * m_plotL3IP2ndTrack2dSig;
  MonitorElement * m_plotL3IP2ndTrack3d;
  MonitorElement * m_plotL3IP2ndTrack3dSig;
  MonitorElement * m_plotL3IP3ndTrack2d;
  MonitorElement * m_plotL3IP3ndTrack2dSig;
  MonitorElement * m_plotL3IP3ndTrack3d;
  MonitorElement * m_plotL3IP3ndTrack3dSig;
  MonitorElement * m_plotL3Discriminator;
};

#endif // DQM_HLTEvF_HLTMonBTagIPSource_H
