// -*- C++ -*-
//
// Package:    DumpSimGeometry
// Class:      DumpSimGeometry
// 
/**\class DumpSimGeometry DumpSimGeometry.cc Reve/DumpSimGeometry/src/DumpSimGeometry.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris D Jones
//         Created:  Wed Sep 26 08:27:23 EDT 2007
//
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "Fireworks/Geometry/interface/DisplayGeomRecord.h"

#include "TGeoManager.h"
#include "TGeoMatrix.h"

#include "TFile.h"
#include "TError.h"
#include "TSystem.h"

//
// class declaration
//

class DumpSimGeometry : public edm::EDAnalyzer
{
public:
  explicit DumpSimGeometry(const edm::ParameterSet&);
  ~DumpSimGeometry();

private:

  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

   std::string m_tag;
   std::string m_outputFileName;
};


//
// constructors and destructor
//
DumpSimGeometry::DumpSimGeometry(const edm::ParameterSet& ps)
{
   m_tag =  ps.getUntrackedParameter<std::string>("tag", "unknown");
   m_outputFileName = ps.getUntrackedParameter<std::string>("outputFileName", "cmsSimGeom.root");

}


DumpSimGeometry::~DumpSimGeometry()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


// ------------ method called to for each event  ------------
void
DumpSimGeometry::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   std::cout << "In the DumpSimGeometry::analyze method..." << std::endl;
   using namespace edm;

   ESTransientHandle<TGeoManager> geoh;
   iSetup.get<DisplayGeomRecord>().get(geoh);
   const TGeoManager *geom = geoh.product(); // const_cast<TGeoManager*>(geoh.product());

   int level = 1 + geom->GetTopVolume()->CountNodes(100, 3);

   std::cout << "In the DumpSimGeometry::analyze method...obtained main geometry, level="
             << level << std::endl;
   
   // TFile f(TString::Format("cmsSimGeom-%d.root", level), "RECREATE");
   TFile f(m_outputFileName.c_str(), "RECREATE");
   f.WriteTObject(geom);
   f.WriteTObject(new TNamed("CMSSW_VERSION", gSystem->Getenv( "CMSSW_VERSION" )));
   f.WriteTObject(new TNamed("tag", m_tag.c_str()));
   f.Close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(DumpSimGeometry);
