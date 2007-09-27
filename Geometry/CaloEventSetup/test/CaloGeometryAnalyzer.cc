// -*- C++ -*-
//
// Package:    CaloGeometryAnalyzer
// Class:      CaloGeometryAnalyzer
// 
/**\class CaloGeometryAnalyzer CaloGeometryAnalyzer.cc test/CaloGeometryAnalyzer/src/CaloGeometryAnalyzer.cc

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
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/CaloGeometry/interface/PreshowerStrip.h"
#include "Geometry/EcalPreshowerAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include <fstream>
#include <iomanip>

//
// class decleration
//

class CaloGeometryAnalyzer : public edm::EDAnalyzer {
   public:
      explicit CaloGeometryAnalyzer( const edm::ParameterSet& );
      ~CaloGeometryAnalyzer();


      virtual void analyze( const edm::Event&, const edm::EventSetup& );
   private:
      // ----------member data ---------------------------
  void build(const CaloGeometry& cg, DetId::Detector det, int subdetn, const char* name);
  int pass_;
  //  bool fullEcalDump_;
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
CaloGeometryAnalyzer::CaloGeometryAnalyzer( const edm::ParameterSet& iConfig )
{
   //now do what ever initialization is needed
  pass_=0;
  //  fullEcalDump_=iConfig.getUntrackedParameter<bool>("fullEcalDump",false);
}


CaloGeometryAnalyzer::~CaloGeometryAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


void CaloGeometryAnalyzer::build(const CaloGeometry& cg, DetId::Detector det, int subdetn, const char* name) {
  std::fstream f(name,std::ios_base::out);
  const CaloSubdetectorGeometry* geom=cg.getSubdetectorGeometry(det,subdetn);

  f << "{" << std::endl;
  f << "  TGeoManager* geoManager = new TGeoManager(\"ROOT\", \"" << name << "\");" << std::endl;
  f << "  TGeoMaterial* dummyMaterial = new TGeoMaterial(\"Vacuum\", 0,0,0); " << std::endl;
  f << "  TGeoMedium* dummyMedium =  new TGeoMedium(\"Vacuum\",1,dummyMaterial);" << std::endl;
  f << "  TGeoVolume* world=geoManager->MakeBox(\"world\",dummyMedium, 8000.0, 8000.0, 14000.0); " << std::endl;
  f << "  geoManager->SetTopVolume(world); " << std::endl;
  f << "  TGeoVolume* box; " << std::endl;
  int n=0;
  std::vector<DetId> ids=geom->getValidDetIds(det,subdetn);
  for (std::vector<DetId>::iterator i=ids.begin(); i!=ids.end(); i++) {
    n++;
    const CaloCellGeometry* cell=geom->getGeometry(*i);
    if (det == DetId::Ecal)
      {
	if (subdetn == EcalBarrel)
	  {
	     const EBDetId did ( *i ) ;
	     const int iea ( did.ietaAbs() ) ;
	     const int ie  ( did.ieta() ) ;
	     const int ip  ( did.iphi() ) ;
	     if( ( iea == 1 || iea == 85 ) &&
		 ( ip  == 1 || ip  == 20 )  )
	     {
		const CaloCellGeometry::CornersVec& co ( cell->getCorners() ) ;
		std::cout << "ieta="<<ie<<", iphi="<<ip
			  <<", "<<std::fixed<<std::setw(8)<<std::setprecision(3)
			  << " ("
			  <<cell->getPosition().x()<<","
			  <<cell->getPosition().y()<<"," 
			  <<cell->getPosition().z()<<")" ;
		for( unsigned int j ( 0 ) ; j != co.size() ; ++j )
		{
		   std::cout<<" ("
			    <<co[j].x()<<","
			    <<co[j].y()<<","
			    <<co[j].z()<<")";
		}
		std::cout<<std::endl;
	     }
	    f << "  // " << EBDetId(*i) << std::endl;
	    
	    f << "  // Checking getClosestCell for position " << dynamic_cast<const TruncatedPyramid*>(cell)->getPosition(0.) << std::endl;
	    EBDetId closestCell = EBDetId(geom->getClosestCell(dynamic_cast<const TruncatedPyramid*>(cell)->getPosition(0.))) ;
	    f << "  // Return position is " << closestCell << std::endl;
	    assert (closestCell == EBDetId(*i) );
	  }
	if (subdetn == EcalEndcap)
	  {
	     const EEDetId did ( *i ) ;
	     const int ix ( did.ix() ) ;
	     const int iy ( did.iy() ) ;
	     if( ( ( ix == 50 || ix == 51 ) &&
		   ( iy ==  5 || iy == 95 )  ) ||
		 ( ( iy == 50 || iy == 51 ) &&
		   ( ix ==  5 || ix == 95 )  )    )

	     {
		const CaloCellGeometry::CornersVec& co ( cell->getCorners() ) ;
		std::cout << "ix="<<ix<<", iy="<<iy
			  <<", "<<std::fixed<<std::setw(8)<<std::setprecision(3)
			  << " ("
			  <<cell->getPosition().x()<<","
			  <<cell->getPosition().y()<<"," 
			  <<cell->getPosition().z()<<")" ;
		for( unsigned int j ( 0 ) ; j != co.size() ; ++j )
		{
		   std::cout<<" ("
			    <<co[j].x()<<","
			    <<co[j].y()<<","
			    <<co[j].z()<<")";
		}
		std::cout<<std::endl;
	     }
	    f << "  // " << EEDetId(*i) << std::endl;
	    f << "  // Checking getClosestCell for position " << dynamic_cast<const TruncatedPyramid*>(cell)->getPosition(0.) << std::endl;
	    EEDetId closestCell= EEDetId(geom->getClosestCell(dynamic_cast<const TruncatedPyramid*>(cell)->getPosition(0.)));
	    f << "  // Return position is " << closestCell << std::endl;
	    assert (closestCell == EEDetId(*i) );
	  }
	if (subdetn == EcalPreshower) 
	  {
	    f << "  // " << ESDetId(*i) << std::endl;
	    f << "  // Checking getClosestCell for position " << cell->getPosition() << " in plane " << ESDetId(*i).plane() << std::endl;
	    ESDetId closestCell=ESDetId((dynamic_cast<const EcalPreshowerGeometry*>(geom))->getClosestCellInPlane(cell->getPosition(),ESDetId(*i).plane()));
	    f << "  // Return position is " << closestCell << std::endl;
	    //sanity checks
            int o_zside = ESDetId(*i).zside();
            int o_plane = ESDetId(*i).plane();
            int o_six   = ESDetId(*i).six();
            int o_siy   = ESDetId(*i).siy();
            int o_strip = ESDetId(*i).strip();

            assert ((o_six <= 20 && cell->getPosition().x() < 0.) || (o_six > 20 && cell->getPosition().x() > 0.));
            assert ((o_siy <= 20 && cell->getPosition().y() < 0.) || (o_siy > 20 && cell->getPosition().y() > 0.));
            assert ((o_zside < 0 && cell->getPosition().z() < 0.) || (o_zside > 0 && cell->getPosition().z() > 0.));
	    assert (closestCell == ESDetId(*i) );
	  }
      }
    else if (det == DetId::Hcal)
      {
	f << "  // " << HcalDetId(*i) << std::endl;
      }
    
    if (det == DetId::Hcal && subdetn==HcalForward) 
      f << "  box=geoManager->MakeBox(\"point\",dummyMedium,1.0,1.0,1.0);" << std::endl;
    else
      f << "  box=geoManager->MakeBox(\"point\",dummyMedium,3.0,3.0,3.0);" << std::endl;
    f << "  world->AddNode(box,"<< n << ",new TGeoHMatrix(TGeoTranslation(" << 
      cell->getPosition().x() << "," << cell->getPosition().y() << "," << cell->getPosition().z() << ")));" << std::endl;
    //   f << (HcalDetId)(*i) << " " << cell->getPosition() << std::endl;
  }
  f << "  geoManager->CloseGeometry();" << std::endl;
  f << "world->Voxelize(\"\"); // now the new geometry is valid for tracking, so you can do \n // even raytracing \n //  if (!canvas) { \n    TCanvas* canvas=new TCanvas(\"EvtDisp\",\"EvtDisp\",500,500); \n //  } \n  canvas->Modified(); \n  canvas->Update();      \n  world->Draw(); \n";
  f << "}" << std::endl;
  f.close();
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
CaloGeometryAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;
   
   std::cout << "Here I am " << std::endl;

   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<IdealGeometryRecord>().get(pG);     
   //
   // get the ecal & hcal geometry
   //
   if (pass_==0) {
     build(*pG,DetId::Ecal,EcalBarrel,"eb.C");
     build(*pG,DetId::Ecal,EcalEndcap,"ee.C");
     //Test eeGetClosestCell in Florian Point
     std::cout << "Checking getClosestCell for position" << GlobalPoint(-38.9692,-27.5548,-317) << std::endl;
     std::cout << "Position of Closest Cell in EE " << dynamic_cast<const TruncatedPyramid*>(pG->getGeometry(EEDetId((*pG).getSubdetectorGeometry(DetId::Ecal,EcalEndcap)->getClosestCell(GlobalPoint(-38.9692,-27.5548,-317)))))->getPosition(0.) << std::endl;
     build(*pG,DetId::Ecal,EcalPreshower,"es.C");
     build(*pG,DetId::Hcal,HcalBarrel,"hb.C");
     build(*pG,DetId::Hcal,HcalEndcap,"he.C");
     build(*pG,DetId::Hcal,HcalOuter,"ho.C");
     build(*pG,DetId::Hcal,HcalForward,"hf.C");
     
   }

   pass_++;
      
}

//define this as a plug-in
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CaloGeometryAnalyzer);
