// -*- C++ -*-
//
// Package:    testCaloCaloGeometryTools
// Class:      testCaloCaloGeometryTools
// 
/**\class testCaloCaloGeometryTools testCaloGeometryTools.cc test/testCaloGeometryTools/src/testCaloGeometryTools.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"
#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "FastSimulation/CaloGeometryTools/interface/Crystal.h"
#include "FastSimulation/Utilities/interface/Histos.h"

#include <TCanvas.h>
#include <TVirtualPad.h>
#include <TStyle.h>
#include <TROOT.h>
#include <TH2F.h>
#include <TH3F.h>
#include <TText.h>
#include <TFile.h>
#include <TArrow.h>
#include <TBox.h>
#include <TPolyLine3D.h>
#include <iostream>
#include <sstream>
//
// class decleration
//

class testCaloGeometryTools : public edm::EDAnalyzer {
public:
  explicit testCaloGeometryTools( const edm::ParameterSet& );
  ~testCaloGeometryTools();
  
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
  void testpoint(const HepPoint3D& , std::string name, bool barrel);
  int pass_;

  Histos * myHistos;
  CaloGeometryHelper myGeometry;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
testCaloGeometryTools::testCaloGeometryTools( const edm::ParameterSet& iConfig )
{
  myHistos = Histos::instance();
  myHistos->book("h100",150,0.,1.5,100,0.,35.);
}


testCaloGeometryTools::~testCaloGeometryTools()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  std::cout << " Writing the histo " << std::endl;
  myHistos->put("Grid.root");
  std::cout << " done " << std::endl;
}

// ------------ method called to produce the data  ------------
void
testCaloGeometryTools::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;
   
   edm::ESHandle<CaloTopology> theCaloTopology;
   iSetup.get<CaloTopologyRecord>().get(theCaloTopology);     

   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<IdealGeometryRecord>().get(pG);     



   // Setup the tools
   myGeometry.setupGeometry(*pG);
   myGeometry.setupTopology(*theCaloTopology);
   myGeometry.initialize();
   
   // Take a point in the barrel
   HepPoint3D p1(129,0.,-50);
   testpoint(p1,"barrel",true);
   HepPoint3D p2(60,60,-317);
   testpoint(p1,"endcap",false);
}


void testCaloGeometryTools::testpoint(const HepPoint3D& point, std::string name, bool barrel)
{
   DetId myCell = myGeometry.getClosestCell(point,true,barrel);
   EcalHitMaker myGrid(&myGeometry,point,myCell,1,7,0);
   
   std::vector<Crystal> myCrystals=myGrid.getCrystals();
   
   float xmin,ymin,zmin,xmax,ymax,zmax;
   xmin=ymin=zmin=99999;
   xmax=ymax=zmax=-9999;
   unsigned nxtal = myCrystals.size();

   std::vector<float> xp,yp,zp;
   
   for(unsigned ic=0;ic<nxtal;++ic)
     {
       HepPoint3D p= myCrystals[ic].getCenter();

       myCrystals[ic].getDrawingCoordinates(xp,yp,zp);
       TPolyLine3D * myxtal= new TPolyLine3D(xp.size(),&xp[0],&yp[0],&zp[0]);
       

       // Build the name of the object
       std::ostringstream oss;
       oss << name << ic;

       myHistos->addObject(oss.str(),myxtal);

       if(xmin > p.x()) xmin=p.x();
       if(ymin > p.y()) ymin=p.y();
       if(zmin > p.z()) zmin=p.z();
       if(xmax < p.x()) xmax=p.x();
       if(ymax < p.y()) ymax=p.y();
       if(zmax < p.z()) zmax=p.z();
     }
   TH3F * frame = new TH3F(std::string(name+"frame").c_str(),"",100,xmin*0.9,xmax*1.1,100,ymin*0.9,ymax*1.1,100,zmin*0.9,zmax*1.1);
   myHistos->addObject("frame"+name,frame);
  
}


//define this as a plug-in
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(testCaloGeometryTools);
