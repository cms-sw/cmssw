#ifndef DTOccupancyTestML_H
#define DTOccupancyTestML_H


/** \class DTOccupancyTest
 * *
 *  DQM Test Client
 *
 *  \author  G. Cerminara - University and INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <DataFormats/MuonDetId/interface/DTLayerId.h>

#include <DQMServices/Core/interface/DQMEDHarvester.h>

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "TH2F.h"

#include <iostream>
#include <string>
#include <map>

class DTGeometry;
class DTChamberId;
class DQMStore;

#include "TFile.h"
#include "TNtuple.h"

class DTOccupancyTestML: public DQMEDHarvester{

public:

  /// Constructor
  DTOccupancyTestML(const edm::ParameterSet& ps);
  
  /// Destructor
  ~DTOccupancyTestML() override;

protected:

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& context) override ;

  /// Endjob
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;
  
  /// DQM Client Diagnostic

  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const &) override;

private:

  /// book the summary histograms
  void bookHistos(DQMStore::IBooker &, const int wheelId, std::string folder, std::string histoTag);

  /// Get the ME name
  std::string getMEName(std::string histoTag, const DTChamberId& chId);

  int getIntegral(TH2F *histo, int, int, int, int, bool);

  // Run the test on the occupancy histos
  int runOccupancyTest(TH2F *histo, const DTChamberId& chId, float& chamberPercentage, tensorflow::GraphDef *graphDef, tensorflow::Session *session);

  std::vector<float> interpolateLayers(std::vector<float> const& inputs, int size, int targetSize);

  std::string topFolder(bool isBooking) const;

  int nevents;

  edm::ESHandle<DTGeometry> muonGeom;

  // wheel summary histograms  
  std::map< int, MonitorElement* > wheelHistos;  
  MonitorElement* summaryHisto;
  MonitorElement* glbSummaryHisto;

  std::set<DTLayerId> monitoredLayers;

  int lsCounter;
  int nMinEvts;
  int nMinEvtsPC;
  int nZeroEvtsPC;

  bool writeRootFile;
  TFile *rootFile;
  TNtuple *ntuple;
  bool tpMode;

  bool runOnAllHitsOccupancies;
  bool runOnNoiseOccupancies;
  bool runOnInTimeOccupancies;
  std::string nameMonitoredHisto;

  bool bookingdone;

  int nChannelTotal;
  int nChannelDead;

};

#endif
