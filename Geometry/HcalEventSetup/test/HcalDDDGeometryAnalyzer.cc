// -*- C++ -*-
//
// Package:    HcalDDDGeometryAnalyzer
// Class:      HcalDDDGeometryAnalyzer
// 
/**\class HcalDDDGeometryAnalyzer HcalDDDGeometryAnalyzer.cc test/HcalDDDGeometryAnalyzer/src/HcalDDDGeometryAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//



// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include <fstream>

//
// class decleration
//

class HcalDDDGeometryAnalyzer : public edm::one::EDAnalyzer<>
{
public:

  explicit HcalDDDGeometryAnalyzer( const edm::ParameterSet& );
  ~HcalDDDGeometryAnalyzer();

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  int pass_;
};

HcalDDDGeometryAnalyzer::HcalDDDGeometryAnalyzer(const edm::ParameterSet& )
{
  pass_=0;
}

HcalDDDGeometryAnalyzer::~HcalDDDGeometryAnalyzer() {}

// ------------ method called to produce the data  ------------
void HcalDDDGeometryAnalyzer::analyze(const edm::Event& , 
				      const edm::EventSetup& iSetup) {

  LogDebug("HCalGeom") << "HcalDDDGeometryAnalyzer::analyze at pass " << pass_;

  edm::ESHandle<CaloGeometry> geometry;
  iSetup.get<CaloGeometryRecord>().get(geometry);     
  //
  // get the ecal & hcal geometry
  //
  if (pass_==0) {
     const std::vector<DetId>& hbCells = geometry->getValidDetIds(DetId::Hcal, 
								  HcalBarrel);
     const std::vector<DetId>& heCells = geometry->getValidDetIds(DetId::Hcal, 
								  HcalEndcap);
     const std::vector<DetId>& hoCells = geometry->getValidDetIds(DetId::Hcal,
								  HcalOuter);
     const std::vector<DetId>& hfCells = geometry->getValidDetIds(DetId::Hcal,
								  HcalForward);
    LogDebug("HCalGeom") << "HcalDDDGeometryAnalyzer:: Hcal Barrel ("
			 << HcalBarrel << ") with " << hbCells.size() 
			 << " valid cells; Hcal Endcap (" << HcalEndcap
			 << ") with " << heCells.size() << " valid cells; "
			 << "Hcal Outer (" << HcalOuter << ") with "
			 << hoCells.size() << " valid cells; and Hcal Forward"
			 << " (" << HcalForward << ") with " << hfCells.size() 
			 << " valid cells";
  }

  pass_++;
      
}

//define this as a plug-in

DEFINE_FWK_MODULE(HcalDDDGeometryAnalyzer);
