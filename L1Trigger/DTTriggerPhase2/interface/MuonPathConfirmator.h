#ifndef L1Trigger_DTTriggerPhase2_MuonPathConfirmator_h
#define L1Trigger_DTTriggerPhase2_MuonPathConfirmator_h

#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/DTTriggerPhase2/interface/constants.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"

#include <iostream>
#include <fstream>

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

// ===============================================================================
// Class declarations
// ===============================================================================

class MuonPathConfirmator {
public:
  // Constructors and destructor
  MuonPathConfirmator(const edm::ParameterSet &pset, edm::ConsumesCollector &iC);
  ~MuonPathConfirmator();

  // Main methods
  void initialise(const edm::EventSetup &iEventSetup);
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           std::vector<cmsdt::metaPrimitive> inMetaPrimitives,
           edm::Handle<DTDigiCollection> dtdigis,
           std::vector<cmsdt::metaPrimitive> &outMetaPrimitives);

  void finish();
  // Other public methods

private:
  // Private methods
  void analyze(cmsdt::metaPrimitive mp,
               edm::Handle<DTDigiCollection> dtdigis,
               std::vector<cmsdt::metaPrimitive> &outMetaPrimitives);
  // Private attributes
  bool debug_;
  double minx_match_2digis_;
  //shift
  edm::FileInPath shift_filename_;
  std::map<int, float> shiftinfo_;
  edm::FileInPath maxdrift_filename_;
  int maxdriftinfo_[5][4][14];
  int max_drift_tdc = -1;

  int PARTIALS_PRECISSION = 4;
  int SEMICHAMBER_H_PRECISSION = 13 + PARTIALS_PRECISSION;
  int SEMICHAMBER_RES_SHR = SEMICHAMBER_H_PRECISSION;
  int LYRANDAHALF_RES_SHR = 4;
  float SEMICHAMBER_H_REAL = ((235. / 2.) / (16. * 6.5)) * std::pow(2, SEMICHAMBER_H_PRECISSION);
  int SEMICHAMBER_H = int(SEMICHAMBER_H_REAL);
};

#endif
