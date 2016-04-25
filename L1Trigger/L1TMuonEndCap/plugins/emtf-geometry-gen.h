#ifndef CSCTrackFinder_slhc_geometry_h
#define CSCTrackFinder_slhc_geometry_h

/**
 * \author L. Gray 6/17/06
 *
 */

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

//ROOT
#include <TH1F.h>
#include <TH2D.h>
#include <TH1I.h>
#include <TFile.h>
#include <TTree.h>
#include <TStyle.h>
#include <TCanvas.h>




#include <L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h>
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"
#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>




class slhc_geometry : public edm::EDAnalyzer {
 public:
  explicit slhc_geometry(edm::ParameterSet const& conf);
  virtual ~slhc_geometry() {}
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  virtual void endJob();
  virtual void beginJob(edm::EventSetup const&);
//   double getTheta(const unsigned& wireGroup, const unsigned& strip, const unsigned& cscId) const;
  double getTheta
  (
      const unsigned& endcap, 
      const unsigned& station, 
      const unsigned& sector, 
      const unsigned& subsector, 
      const unsigned& wireGroup, 
      const unsigned& strip, 
      const unsigned& cscId,
      edm::EventSetup const& es
  ) const;

  double getTheta_limited
  (
      const unsigned& endcap,
      const unsigned& station,
      const unsigned& sector,
      const unsigned& subsector,
      const unsigned& wireGroup,
      const unsigned& strip,
      const unsigned& cscId,
      edm::EventSetup const& es
  ) const;

  double getTheta_wire(const unsigned& endcap, const unsigned& station, const unsigned& sector, const unsigned& subsector, const unsigned& wire, const unsigned& strip, const unsigned& cscId) const;
  
  double getGlobalPhiValue
  (
      const unsigned& endcap, 
      const unsigned& station, 
      const unsigned& sector, 
      const unsigned& subsector, 
      const unsigned& wireGroup, 
      const unsigned& strip, 
      const unsigned& cscId, 
      edm::EventSetup const& es
   ) const;
  
//   double getLocalPhiValue(const unsigned& endcap, const unsigned& station, const unsigned& sector, const unsigned& subsector, const unsigned& wireGroup, const unsigned& strip, const unsigned& cscId) const;
  
  double getLocalSectorPhiValue(const unsigned& endcap, const unsigned& station, const unsigned& sector, const unsigned& subsector, const unsigned& wireGroup, const unsigned& strip, const unsigned& cscId, edm::EventSetup const& es) const;
  
  void generateLUTs(int endcap, edm::EventSetup const& es);
  void generateLUTStation1(int endcap, edm::EventSetup const& es);
  void get_ring_chamber
    (
     const unsigned& station, 
     const unsigned& sector, 
     const unsigned& subsector, 
     const unsigned& cscId, 
     unsigned& ring, 
     unsigned& ichamber
     ) const;

  double get_sector_phi_hs 
    (
     const unsigned& endcap, 
     const unsigned& station, 
     const unsigned& sector, 
     const unsigned& subsector, 
     const unsigned& wireGroup, 
     const unsigned& halfstrip, 
     const unsigned& cscId, 
          const bool& nb,
     edm::EventSetup const& es
     ) const;

  // [sector][station][chamber]
  // chamber includes CSCIDs 10,11,12 for ME1/1A
  int num_of_wiregroups[6][5][13];
  int num_of_strips[6][5][13];
  double strip_phi_pitch[6][5][13];
  double strip_dphi[6][5][13];
 
 private:
  edm::ParameterSet lutParam;


  
};

DEFINE_FWK_MODULE(slhc_geometry);

#endif
