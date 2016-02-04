#ifndef DTOccupancyTest_H
#define DTOccupancyTest_H


/** \class DTOccupancyTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2011/06/10 13:50:12 $
 *  $Revision: 1.9 $
 *  \author  G. Cerminara - University and INFN Torino
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include "DQMServices/Core/interface/MonitorElement.h"
#include <DataFormats/MuonDetId/interface/DTLayerId.h>

#include "TH2F.h"

#include <iostream>
#include <string>
#include <map>

class DTGeometry;
class DTChamberId;
class DQMStore;

#include "TFile.h"
#include "TNtuple.h"

class DTOccupancyTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTOccupancyTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTOccupancyTest();

protected:

  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& context) ;

  /// Endjob
  void endJob();
  
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context);


  /// Analyze
  void analyze(const edm::Event& event, const edm::EventSetup& context);

private:

  /// book the summary histograms
  void bookHistos(const int wheelId, std::string folder, std::string histoTag);

  /// Get the ME name
  std::string getMEName(std::string histoTag, const DTChamberId& chId);

  // Run the test on the occupancy histos
  int runOccupancyTest(TH2F *histo, const DTChamberId& chId, float& chamberPercentage);

  std::string topFolder() const;

  int nevents;

  DQMStore* dbe;

  edm::ESHandle<DTGeometry> muonGeom;

  // wheel summary histograms  
  std::map< int, MonitorElement* > wheelHistos;  
  MonitorElement* summaryHisto;
  MonitorElement* glbSummaryHisto;

  std::set<DTLayerId> monitoredLayers;

  int lsCounter;
  int nMinEvts;

  bool writeRootFile;
  TFile *rootFile;
  TNtuple *ntuple;
  bool tpMode;

  bool runOnAllHitsOccupancies;
  bool runOnNoiseOccupancies;
  bool runOnInTimeOccupancies;
  std::string nameMonitoredHisto;

};

#endif
