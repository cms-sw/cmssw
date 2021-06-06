#ifndef CalibMuon_DTCalibration_DTVDriftWriter_h
#define CalibMuon_DTCalibration_DTVDriftWriter_h

/*  \class DTVDriftWriter
 *  Instantiates configurable algo plugin to
 *  compute and write vDrift DB.
 * 
 *  Author of original version: M. Giunta
 *  \author A. Vilela Pereira 
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <string>

class DTMtime;
class DTRecoConditions;
class DTGeometry;
namespace dtCalibration {
  class DTVDriftBaseAlgo;
}

class DTVDriftWriter : public edm::EDAnalyzer {
public:
  DTVDriftWriter(const edm::ParameterSet& pset);
  ~DTVDriftWriter() override;

  // Operations
  void beginRun(const edm::Run& run, const edm::EventSetup& setup) override;
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override {}
  void endJob() override;

private:
  std::string granularity_;  // enforced by SL

  const DTMtime* mTimeMap_;            // legacy DB object
  const DTRecoConditions* vDriftMap_;  // DB object in new format
  bool readLegacyVDriftDB;             // which format to use to read old values
  bool writeLegacyVDriftDB;            // which format to be created

  edm::ESHandle<DTGeometry> dtGeom_;

  std::unique_ptr<dtCalibration::DTVDriftBaseAlgo> vDriftAlgo_;
};
#endif
