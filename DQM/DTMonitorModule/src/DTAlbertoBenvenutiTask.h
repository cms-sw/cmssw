#ifndef DTAlbertoBenvenutiTask_H
#define DTAlbertoBenvenutiTask_H


/*
 * \file DTAlbertoBenvenutiTask.h
 *
 * $Date: 2012/09/24 16:08:06 $
 * $Revision: 1.5 $
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
  void beginJob();

  void beginRun(const edm::Run&, const edm::EventSetup&);

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
  float kFactor;

  edm::ParameterSet parameters;

  edm::ESHandle<DTGeometry> muonGeom;

  edm::ESHandle<DTTtrig> tTrigMap;
  edm::ESHandle<DTT0> t0Map;

  std::string outputFile;

  std::map<DTWireId, TH1F*> TBMap;


};

#endif
