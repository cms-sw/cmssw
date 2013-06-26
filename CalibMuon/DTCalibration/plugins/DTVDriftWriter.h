#ifndef CalibMuon_DTCalibration_DTVDriftWriter_h
#define CalibMuon_DTCalibration_DTVDriftWriter_h

/*  \class DTVDriftWriter
 *  Instantiates configurable algo plugin to
 *  compute and write vDrift DB.
 * 
 *  $Date: 2010/11/17 17:54:23 $
 *  $Revision: 1.2 $
 *  Author of original version: M. Giunta
 *  \author A. Vilela Pereira 
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <string>

class DTMtime;
class DTGeometry;
class DTVDriftBaseAlgo;

class DTVDriftWriter : public edm::EDAnalyzer {
public:
  DTVDriftWriter(const edm::ParameterSet& pset);
  virtual ~DTVDriftWriter();

  // Operations
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& setup);
  virtual void analyze(const edm::Event & event, const edm::EventSetup& eventSetup) {}
  virtual void endJob();
 
private:
  std::string granularity_; // enforced by SL

  const DTMtime* mTimeMap_;
  edm::ESHandle<DTGeometry> dtGeom_;

  DTVDriftBaseAlgo* vDriftAlgo_; 
};
#endif

