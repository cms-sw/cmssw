#ifndef MuonAlignmentSummary_H
#define MuonAlignmentSummary_H

/** \class MuonAlignmentSummary
 *
 *  DQM client for muon alignment summary
 *
 *  \author J. Fernandez - Univ. Oviedo <Javier.Fernandez@cern.ch>
 */

#include <cmath>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

namespace edm {
  class ParameterSet;
  class EventSetup;
  class InputTag;
}  // namespace edm

class TH1F;

class MuonAlignmentSummary : public DQMEDHarvester {
public:
  /// Constructor
  MuonAlignmentSummary(const edm::ParameterSet &);

  /// Destructor
  ~MuonAlignmentSummary() override;

  // Book histograms
  void dqmEndJob(DQMStore::IBooker &,
                 DQMStore::IGetter &) override;  // performed in the endJob

private:
  // ----------member data ---------------------------

  MonitorElement *hLocalPositionDT;
  MonitorElement *hLocalPositionRmsDT;
  MonitorElement *hLocalAngleDT;
  MonitorElement *hLocalAngleRmsDT;

  MonitorElement *hLocalXMeanDT;
  MonitorElement *hLocalXRmsDT;
  MonitorElement *hLocalYMeanDT;
  MonitorElement *hLocalYRmsDT;
  MonitorElement *hLocalPhiMeanDT;
  MonitorElement *hLocalPhiRmsDT;
  MonitorElement *hLocalThetaMeanDT;
  MonitorElement *hLocalThetaRmsDT;

  MonitorElement *hLocalPositionCSC;
  MonitorElement *hLocalPositionRmsCSC;
  MonitorElement *hLocalAngleCSC;
  MonitorElement *hLocalAngleRmsCSC;

  MonitorElement *hLocalXMeanCSC;
  MonitorElement *hLocalXRmsCSC;
  MonitorElement *hLocalYMeanCSC;
  MonitorElement *hLocalYRmsCSC;
  MonitorElement *hLocalPhiMeanCSC;
  MonitorElement *hLocalPhiRmsCSC;
  MonitorElement *hLocalThetaMeanCSC;
  MonitorElement *hLocalThetaRmsCSC;

  edm::ParameterSet parameters;

  // Switch for verbosity
  std::string metname;

  // mean and rms histos ranges
  double meanPositionRange, rmsPositionRange, meanAngleRange, rmsAngleRange;

  // flags to decide on subdetector and summary histograms
  bool doDT, doCSC;

  // Top folder in root file
  std::string MEFolderName;
  std::stringstream topFolder;
};
#endif
