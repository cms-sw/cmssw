#ifndef DTTTrigCorrectionFirst_H
#define DTTTrigCorrectionFirst_H

/** \class DTTTrigCorrection
 *  Class which read a ttrig DB and correct it with
 *  the near SL (or the global average)
 *
 *  \author S. Maselli - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <string>

class DTTtrig;
class DTGeometry;

class DTTTrigCorrectionFirst : public edm::EDAnalyzer {
public:
  /// Constructor
  DTTTrigCorrectionFirst(const edm::ParameterSet& pset);

  /// Destructor
  ~DTTTrigCorrectionFirst() override;

  // Operations

  void beginJob() override {}
  void beginRun( const edm::Run& run, const edm::EventSetup& setup ) override;
  void analyze(const edm::Event& event, const edm::EventSetup& setup) override{}

  void endJob() override;

protected:

private:
  const DTTtrig *tTrigMap;
  edm::ESHandle<DTGeometry> muonGeom;

  std::string dbLabel;

  bool debug;
  double ttrigMin,ttrigMax,rmsLimit; 
};
#endif

