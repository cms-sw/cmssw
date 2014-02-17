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
// $Id: DumpSimGeometry.cc,v 1.3 2012/08/01 04:09:50 amraktad Exp $
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

//
// class declaration
//

class DumpSimGeometry : public edm::EDAnalyzer
{
public:
  explicit DumpSimGeometry(const edm::ParameterSet&);
  ~DumpSimGeometry();

private:
  // virtual void beginJob();
  // virtual void endJob();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
};


//
// constructors and destructor
//
DumpSimGeometry::DumpSimGeometry(const edm::ParameterSet&)
{
   // now do what ever initialization is needed
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
   
   TFile f(TString::Format("cmsSimGeom-%d.root", level), "RECREATE");
   f.WriteTObject(geom);
   f.Close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(DumpSimGeometry);
