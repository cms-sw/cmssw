#ifndef Phase2L1Trigger_DTTrigger_MuonPathAssociator_h
#define Phase2L1Trigger_DTTrigger_MuonPathAssociator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"

#include "L1Trigger/DTTriggerPhase2/interface/MuonPath.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"
#include "L1Trigger/DTTriggerPhase2/interface/GlobalCoordsObtainer.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include <iostream>
#include <fstream>

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

// ===============================================================================
// Class declarations
// ===============================================================================

class MuonPathAssociator {
public:
  // Constructors and destructor
  MuonPathAssociator(const edm::ParameterSet &pset, edm::ConsumesCollector &iC,
    std::shared_ptr<GlobalCoordsObtainer> & globalcoordsobtainer);
  ~MuonPathAssociator();

  // Main methods
  void initialise(const edm::EventSetup &iEventSetup);
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           edm::Handle<DTDigiCollection> digis,
           std::vector<cmsdt::metaPrimitive> &inMPaths,
           std::vector<cmsdt::metaPrimitive> &outMPaths);

  void finish();

  // Other public methods

  bool shareFit(cmsdt::metaPrimitive first, cmsdt::metaPrimitive second);
  bool isNotAPrimo(cmsdt::metaPrimitive first, cmsdt::metaPrimitive second);
  void removeSharingFits(std::vector<cmsdt::metaPrimitive> &chamberMPaths,
                         std::vector<cmsdt::metaPrimitive> &allMPaths);
  void removeSharingHits(std::vector<cmsdt::metaPrimitive> &firstMPaths,
                         std::vector<cmsdt::metaPrimitive> &secondMPaths,
                         std::vector<cmsdt::metaPrimitive> &allMPaths);
  void printmPC(cmsdt::metaPrimitive mP);

  // Public attributes
  DTGeometry const *dtGeo_;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomH_;

private:
  // Private methods
  void correlateMPaths(edm::Handle<DTDigiCollection> digis,
                       std::vector<cmsdt::metaPrimitive> &inMPaths,
                       std::vector<cmsdt::metaPrimitive> &outMPaths);

  bool hasPosRF(int wh, int sec) { return wh > 0 || (wh == 0 && sec % 4 > 1); }

  // Private attributes
  bool debug_;
  bool clean_chi2_correlation_;
  bool useBX_correlation_;
  bool allow_confirmation_;
  double dT0_correlate_TP_;
  double dBX_correlate_TP_;
  double dTanPsi_correlate_TP_;
  double minx_match_2digis_;
  double chi2corTh_;
  bool cmssw_for_global_;
  std::string geometry_tag_;

  //shift
  edm::FileInPath shift_filename_;
  std::map<int, float> shiftinfo_;
  
  // global coordinates
  std::shared_ptr<GlobalCoordsObtainer> globalcoordsobtainer_;
};

#endif
