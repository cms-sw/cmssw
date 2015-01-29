#ifndef DTCreateSummaryHistos_H
#define DTCreateSummaryHistos_H


/** \class DTCreateSummaryHistos
 * *
 *  DQM Test Client
 *
 *  \author  G. Mila - INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah ncpp-um-my
 *
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

#include <DQMServices/Core/interface/DQMEDHarvester.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include "TPostScript.h"

class DTGeometry;

//-class DTCreateSummaryHistos: public edm::EDAnalyzer{
class DTCreateSummaryHistos: public DQMEDHarvester{

public:

  /// Constructor
  DTCreateSummaryHistos(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTCreateSummaryHistos();

protected:
                                                                                                       
  /// BeginRun
   void beginRun(const edm::Run& run, const edm::EventSetup& setup);

   void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

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
  
  
