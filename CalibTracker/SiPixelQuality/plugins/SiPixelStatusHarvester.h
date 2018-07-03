#ifndef SiPixelStatusHarvester_H
#define SiPixelStatusHarvester_H

/** \class SiPixelStatusHarvester
 *  harvest per-lumi prduced SiPixelDetector status and make the payload for SiPixelQualityFromDB
 *
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibTracker/SiPixelQuality/interface/SiPixelStatusManager.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"

class SiPixelStatusHarvester : public edm::EDAnalyzer {
 public:

  // Constructor
  SiPixelStatusHarvester(const edm::ParameterSet&);

  // Destructor
  ~SiPixelStatusHarvester() override;
  
  // Operations
  void beginJob            () override;
  void endJob              () override;  
  void analyze             (const edm::Event&          , const edm::EventSetup&) override;
  void beginRun            (const edm::Run&            , const edm::EventSetup&) override;
  void endRun              (const edm::Run&            , const edm::EventSetup&) override;
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  void endLuminosityBlock  (const edm::LuminosityBlock&, const edm::EventSetup&) override;

 protected:

 private:
  // Parameters
  std::string outputBase_;
  int aveDigiOcc_;
  int nLumi_;
  std::string moduleName_;
  std::string label_;  
  // harvest helper classs that setup the IOV structure
  SiPixelStatusManager siPixelStatusManager_;
  // debug mode
  bool debug_;
  // for DB output naming
  std::string recordName_;

  // permanent known bad components
  const SiPixelQuality* badPixelInfo_;

  // last lumi section of the SiPixeDetectorStatus data
  edm::LuminosityBlockNumber_t endLumiBlock_;

  // "step function" for IOV
  edm::LuminosityBlockNumber_t stepIOV(edm::LuminosityBlockNumber_t pin, std::map<edm::LuminosityBlockNumber_t,edm::LuminosityBlockNumber_t> IOV);

};

#endif
