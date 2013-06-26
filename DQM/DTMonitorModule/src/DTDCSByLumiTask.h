#ifndef DTDCSByLumiTask_H
#define DTDCSByLumiTask_H

/*
 * \file DTDCSByLumiTask.h
 *
 * $Date: 2011/03/02 13:56:39 $
 * $Revision: 1.1 $
 * \author C. Battilana - CIEMAT
 * \author P. Bellan - INFN PD
 * \author A. Branca = INFN PD
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <FWCore/Framework/interface/LuminosityBlock.h>

#include <vector>

class DTGeometry;
class DQMStore;
class MonitorElement;
class DTHVStatus;

class DTDCSByLumiTask: public edm::EDAnalyzer{

public:

  /// Constructor
  DTDCSByLumiTask(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTDCSByLumiTask();

protected:

  /// BeginJob
  void beginJob();

  /// Begin Run
  void beginRun(const edm::Run&, const edm::EventSetup&);

  /// Book Monitor Elements
  void bookHistos();

  /// By Lumi DCS DB Operation
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// By Lumi DCS DB Operation
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& setup);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

private:
  
  std::string topFolder() const;

  int theEvents;
  int theLumis;

  bool DTHVRecordFound;

  DQMStore* theDQMStore;
  edm::ESHandle<DTGeometry> theDTGeom;
  // CB Get HV DB and loop on half layers
  edm::ESHandle<DTHVStatus> hvStatus;

  std::vector<MonitorElement*> hActiveUnits;

};

#endif
