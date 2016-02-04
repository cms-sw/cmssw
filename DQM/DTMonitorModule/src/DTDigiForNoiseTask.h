#ifndef DTDigiForNoiseTask_H
#define DTDigiForNoiseTask_H

/*
 * \file DTDigiForNoiseTask.h
 *
 * $Date: 2010/01/05 10:14:40 $
 * $Revision: 1.6 $
 * \author G. Mila - INFN Torino
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DTGeometry;
class DTSuperLayerId;
class DTLayerId;
class DTChamberId;


class DTDigiForNoiseTask: public edm::EDAnalyzer{

public:

  /// Constructor
  DTDigiForNoiseTask(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTDigiForNoiseTask();

protected:

  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(const edm::Run&, const edm::EventSetup&);

  /// Book the ME
  void bookHistos(const DTLayerId& dtSL);
 
  /// To reset the MEs
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

private:

  bool debug;
  int nevents;

  DQMStore* dbe;

  edm::ParameterSet parameters;

  edm::ESHandle<DTGeometry> muonGeom;

  std::map< DTLayerId, MonitorElement* > digiHistos;


};

#endif
