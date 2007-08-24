#ifndef DTAlbertoBenvenutiTask_H
#define DTAlbertoBenvenutiTask_H


/*
 * \file DTAlbertoBenvenutiTask.h
 *
 * $Date: 2007/03/22 18:52:01 $
 * $Revision: 1.11 $
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

#include "DataFormats/LTCDigi/interface/LTCDigi.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include "TH1F.h"

class DTGeometry;
class DTWireId;
class DTTtrig;
class DTT0;


class DTAlbertoBenvenutiTask: public edm::EDAnalyzer{

public:

  /// Constructor
  DTAlbertoBenvenutiTask(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTAlbertoBenvenutiTask();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// Book the ME
  void bookHistos(const DTWireId dtWire);
 
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

private:

  bool debug;
  int nevents;

  /// no needs to be precise. Value from PSets will always be used
  int tMax;
  int maxTDCHits;

  /// tTrig from the DB
  float tTrig;
  float tTrigRMS;

  edm::Handle<LTCDigiCollection> ltcdigis;

  edm::ParameterSet parameters;

  edm::ESHandle<DTGeometry> muonGeom;

  edm::ESHandle<DTTtrig> tTrigMap;
  edm::ESHandle<DTT0> t0Map;

  std::string outputFile;

  std::map<DTWireId, TH1F*> TBMap;


};

#endif
