#ifndef DTNoiseAnalysisTest_H
#define DTNoiseAnalysisTest_H

/** \class DTNoiseAnalysisTest
 * *
 *  DQM Test Client
 *
 *  \author  G. Mila - INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 *   
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <iostream>
#include <string>
#include <map>

class DTGeometry;
class DTChamberId;
class DTSuperLayerId;

class DTNoiseAnalysisTest : public DQMEDHarvester {
public:
  /// Constructor
  DTNoiseAnalysisTest(const edm::ParameterSet& ps);

  /// Destructor
  ~DTNoiseAnalysisTest() override;

protected:
  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& context) override;

  /// book the summary histograms

  void bookHistos(DQMStore::IBooker&);

  /// DQM Client Diagnostic
  void dqmEndLuminosityBlock(DQMStore::IBooker&,
                             DQMStore::IGetter&,
                             edm::LuminosityBlock const&,
                             edm::EventSetup const&) override;

  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  /// Get the ME name
  std::string getMEName(const DTChamberId& chID);
  std::string getSynchNoiseMEName(int wheelId) const;

  int nevents;
  int nMinEvts;

  bool bookingdone;

  // the dt geometry
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> muonGeomToken_;
  const DTGeometry* muonGeom;

  // paramaters from cfg
  int noisyCellDef;
  bool isCosmics;
  bool doSynchNoise;
  bool detailedAnalysis;
  double maxSynchNoiseRate;
  double noiseSafetyFactor;

  // wheel summary histograms
  std::map<int, MonitorElement*> noiseHistos;
  std::map<int, MonitorElement*> noisyCellHistos;
  MonitorElement* summaryNoiseHisto;
  MonitorElement* threshChannelsHisto;
  MonitorElement* summarySynchNoiseHisto;
  MonitorElement* glbSummarySynchNoiseHisto;

  //values based on F. Romana research, estimate the background rate per chamber and set a threshold to spot noisy wires with a safety factor
  static constexpr float cellW = 4.2;    //cm
  static constexpr float instLumi = 20;  //E33 cm-2 s-1, reference for Run3
  static constexpr std::array<std::array<float, 4>, 3> kW_MB = {
      {{{0.41, 0.08, 0.01, 0.15}},
       {{0.17, 0.04, 0.01, 0.15}},
       {{0.06, 0.02, 0.01, 0.15}}}};  // in units of E33 cm-2 s-1, 3 wheel types x 4 MB stations
  static constexpr std::array<std::array<float, 4>, 2> lenghtSL_MB = {
      {{{206, 252, 302, 0}}, {{240, 240, 240, 240}}}};  //Theta and Phi SL1 SL3
};

#endif
