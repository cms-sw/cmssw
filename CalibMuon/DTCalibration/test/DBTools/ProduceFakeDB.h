#ifndef ProduceFakeDB_H
#define ProduceFakeDB_H

/** \class ProduceFakeDB
 *  Class which produce fake DB of ttrig,t0,vdrift
 *
 *  $Date: 2010/01/19 09:51:31 $
 *  $Revision: 1.2 $
 *  \author S. Bolognesi - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class DTGeometry;

class ProduceFakeDB : public edm::EDAnalyzer {
public:
  /// Constructor
  ProduceFakeDB(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~ProduceFakeDB();

  // Operations
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& setup );

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){}

  virtual void endJob();

protected:

private:
  edm::ESHandle<DTGeometry> muonGeom;

  std::string dbToProduce;
  edm::ParameterSet ps;
};
#endif

