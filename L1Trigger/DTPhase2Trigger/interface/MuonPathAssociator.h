#ifndef Phase2L1Trigger_DTTrigger_MuonPathAssociator_cc
#define Phase2L1Trigger_DTTrigger_MuonPathAssociator_cc

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
//#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
//#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
//#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
//#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
//
#include "L1Trigger/DTPhase2Trigger/interface/muonpath.h"
#include "L1Trigger/DTPhase2Trigger/interface/analtypedefs.h"
#include "L1Trigger/DTPhase2Trigger/interface/constants.h"

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
  MuonPathAssociator(const edm::ParameterSet& pset);
  virtual ~MuonPathAssociator();
    
  // Main methods
  void initialise(const edm::EventSetup& iEventSetup);
  void run(edm::Event& iEvent, const edm::EventSetup& iEventSetup, edm::Handle<DTDigiCollection> digis,
	   std::vector<metaPrimitive> &inMPaths, std::vector<metaPrimitive> &outMPaths);
  
    
  void finish();
    
  // Other public methods

  bool shareFit (metaPrimitive first, metaPrimitive second); 
  bool isNotAPrimo (metaPrimitive first, metaPrimitive second); 
  void removeSharingFits (std::vector<metaPrimitive> &chamberMPaths,
			  std::vector<metaPrimitive> &allMPaths);
  void removeSharingHits (std::vector<metaPrimitive> &firstMPaths,
			  std::vector<metaPrimitive> &secondMPaths,
			  std::vector<metaPrimitive> &allMPaths);
  void printmPC(metaPrimitive mP);
 
  // Public attributes
 edm::ESHandle<DTGeometry> dtGeo;  


private:
 
  // Private methods
 void correlateMPaths(edm::Handle<DTDigiCollection> digis, std::vector<metaPrimitive> &inMPaths, std::vector<metaPrimitive> &outMPaths);

  //  void associate(metaPrimitive MP, std::vector<metaPrimitive> &outMP);

  bool hasPosRF(int wh,int sec) {    return  wh>0 || (wh==0 && sec%4>1); }
  
  // Private attributes
  Bool_t debug;
  Bool_t clean_chi2_correlation;
  Bool_t useBX_correlation;
  Bool_t allow_confirmation;
  double dT0_correlate_TP;
  double dBX_correlate_TP;
  double dTanPsi_correlate_TP;
  double minx_match_2digis;
  double chi2corTh;
  Bool_t use_LSB;
  double tanPsi_precision; 
  double x_precision; 

  //shift
  edm::FileInPath shift_filename;
  std::map<int,float> shiftinfo;
  
};


#endif
