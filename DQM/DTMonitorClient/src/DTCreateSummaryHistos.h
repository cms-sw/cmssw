#ifndef DTCreateSummaryHistos_H
#define DTCreateSummaryHistos_H


/** \class DTCreateSummaryHistos
 * *
 *  DQM Test Client
 *
 *  $Date: 2010/01/05 10:15:45 $
 *  $Revision: 1.4 $
 *  \author  G. Mila - INFN Torino
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include "TPostScript.h"

class DTGeometry;

class DTCreateSummaryHistos: public edm::EDAnalyzer{

public:

  /// Constructor
  DTCreateSummaryHistos(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTCreateSummaryHistos();

protected:

  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(const edm::Run& run, const edm::EventSetup& setup);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();


 private:

  int nevents;
  std::string MainFolder;

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;

  // The file which contain the occupancy plot and the digi event plot
  TFile *theFile;
  
  // The *.ps file which contains the summary histos
  TPostScript *psFile;
  std::string PsFileName;

  // The histos to write in the *.ps file
  bool DataIntegrityHistos;
  bool DigiHistos;
  bool RecoHistos;
  bool ResoHistos;
  bool EfficiencyHistos;
  bool TestPulsesHistos;
  bool TriggerHistos;
  
  // The DDUId
  int DDUId;
  // The run number
  int runNumber;

};

#endif
  
  
