// -*- C++ -*-
//
// Package:    TrackerTreeGenerator
// Class:      TrackerTreeGenerator
// 
/**\class TrackerTreeGenerator TrackerTreeGenerator.cc Alignment/TrackerAlignment/plugins/TrackerTreeGenerator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Johannes Hauk
//         Created:  Fri Jan 16 14:09:52 CET 2009
//         Modified by: Ajay Kumar (University of Delhi)
//
// $Id: TrackerTreeGenerator.cc,v 1.1 2011/09/01 11:18:08 hauk Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "Alignment/TrackerAlignment/interface/TrackerTreeVariables.h"

#include "TTree.h"
//
// class decleration
//

class TrackerTreeGenerator : public edm::EDAnalyzer {
   public:
      explicit TrackerTreeGenerator(const edm::ParameterSet&);
      ~TrackerTreeGenerator();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      
      const bool createEntryForDoubleSidedModule_;
      std::vector<TrackerTreeVariables> vTkTreeVar_;
      edm::ParameterSet theParameterSet;
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
TrackerTreeGenerator::TrackerTreeGenerator(const edm::ParameterSet& iConfig):
createEntryForDoubleSidedModule_(iConfig.getParameter<bool>("createEntryForDoubleSidedModule")),
theParameterSet( iConfig )
{
}


TrackerTreeGenerator::~TrackerTreeGenerator()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TrackerTreeGenerator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   //iSetup.get<TrackerDigiGeometryRecord>().get(tkGeom);
   // now try to take directly the ideal geometry independent of used geometry in Global Tag
   edm::ESHandle<GeometricDet> geometricDet;
   iSetup.get<IdealGeometryRecord>().get(geometricDet);

   //Retrieve tracker topology from geometry
   edm::ESHandle<TrackerTopology> tTopoHandle;
   iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
   const TrackerTopology* const tTopo = tTopoHandle.product();
   
   edm::ESHandle<PTrackerParameters> ptp;
   iSetup.get<PTrackerParametersRcd>().get(ptp);
   TrackerGeomBuilderFromGeometricDet trackerBuilder;
   TrackerGeometry* tkGeom = trackerBuilder.build(&(*geometricDet), *ptp, tTopo);
 
   const TrackerGeometry *bareTkGeomPtr = &(*tkGeom);
   
   edm::LogInfo("TrackerTreeGenerator") //<< "@SUB=analyze"
                                        << "There are " << bareTkGeomPtr->detIds().size()
					<< " dets and "<<bareTkGeomPtr->detUnitIds().size()
					<<" detUnits in the Geometry Record";
   
   if(createEntryForDoubleSidedModule_)edm::LogInfo("TrackerTreeGenerator") << "Create entry for each module AND one entry for virtual "
                                                                            << "double-sided module in addition";
   else edm::LogInfo("TrackerTreeGenerator") << "Create one entry for each physical module, do NOT create additional entry for virtual "
                                             << "double-sided module";
   
   const TrackingGeometry::DetIdContainer& detIdContainer = bareTkGeomPtr->detIds();
   
   std::vector<DetId>::const_iterator iDet;
   for(iDet = detIdContainer.begin(); iDet != detIdContainer.end(); ++iDet){
     
     const DetId& detId     = *iDet;
     const GeomDet& geomDet = *tkGeom->idToDet(*(&detId));
     const Surface& surface = (&geomDet)->surface();
     
     TrackerTreeVariables tkTreeVar;
     uint32_t rawId = detId.rawId();
     tkTreeVar.rawId    = rawId;
     tkTreeVar.subdetId = detId.subdetId();
     
     LocalPoint lPModule(0.,0.,0.), lUDirection(1.,0.,0.), lVDirection(0.,1.,0.), lWDirection(0.,0.,1.);
     GlobalPoint gPModule    = surface.toGlobal(lPModule),
                 gUDirection = surface.toGlobal(lUDirection),
                 gVDirection = surface.toGlobal(lVDirection),
	         gWDirection = surface.toGlobal(lWDirection);
     double dR(999.), dPhi(999.), dZ(999.);
     if(tkTreeVar.subdetId==PixelSubdetector::PixelBarrel || tkTreeVar.subdetId==StripSubdetector::TIB
                                                          || tkTreeVar.subdetId==StripSubdetector::TOB){
       dR   = gWDirection.perp() - gPModule.perp();
       dPhi = deltaPhi(gUDirection.barePhi(),gPModule.barePhi());
       dZ   = gVDirection.z() - gPModule.z();
       tkTreeVar.uDirection = dPhi>0. ? 1 : -1;
       tkTreeVar.vDirection = dZ>0.   ? 1 : -1;
       tkTreeVar.wDirection = dR>0.   ? 1 : -1;
     }else if(tkTreeVar.subdetId==PixelSubdetector::PixelEndcap){
       dR   = gUDirection.perp() - gPModule.perp();
       dPhi = deltaPhi(gVDirection.barePhi(),gPModule.barePhi());
       dZ   = gWDirection.z() - gPModule.z();
       tkTreeVar.uDirection = dR>0.   ? 1 : -1;
       tkTreeVar.vDirection = dPhi>0. ? 1 : -1;
       tkTreeVar.wDirection = dZ>0.   ? 1 : -1;
     }else if(tkTreeVar.subdetId==StripSubdetector::TID || tkTreeVar.subdetId==StripSubdetector::TEC){
       dR = gVDirection.perp() - gPModule.perp();
       dPhi = deltaPhi(gUDirection.barePhi(),gPModule.barePhi());
       dZ = gWDirection.z() - gPModule.z();
       tkTreeVar.uDirection = dPhi>0. ? 1 : -1;
       tkTreeVar.vDirection = dR>0.   ? 1 : -1;
       tkTreeVar.wDirection = dZ>0.   ? 1 : -1;
     }
     tkTreeVar.posR         = gPModule.perp();
     tkTreeVar.posPhi       = gPModule.barePhi();     // = gPModule.barePhi().degrees();
     tkTreeVar.posEta       = gPModule.eta();
     tkTreeVar.posX         = gPModule.x();
     tkTreeVar.posY         = gPModule.y();
     tkTreeVar.posZ         = gPModule.z();
       
     
     if(tkTreeVar.subdetId == PixelSubdetector::PixelBarrel){
      unsigned int whichHalfBarrel(1);  //DetId does not know about halfBarrels is PXB ...
       // The easiest thing is to get the half barrel from the x-position.
       // This way we can run on both Phase 0 and Phase I without having to change the generator
       if( tkTreeVar.posX < 0. ) whichHalfBarrel=2;
       
       // Hard coded Id ranges. Might be useful in case of very strongly misaligned scenarios
       // Phase 0
       // if( (rawId>=302056964 && rawId<302059300) || (rawId>=302123268 && rawId<302127140) || (rawId>=302189572 && rawId<302194980) )whichHalfBarrel=2;
       // Phase I
       // if( (rawId>=303054852 && rawId<303075370) || (rawId>=304119812 && rawId<304173090) || (rawId>=305184772 && rawId<305270820) || (rawId>=306253828 && rawId<306380840) ) whichHalfBarrel=2;
       
       tkTreeVar.layer  = tTopo->pxbLayer(*(&detId));
       tkTreeVar.half   = whichHalfBarrel;
       tkTreeVar.rod    = tTopo->pxbLadder(*(&detId));     // ... so, ladder is not per halfBarrel-Layer, but per barrel-layer!
       tkTreeVar.module = tTopo->pxbModule(*(&detId));
      }
     else if(tkTreeVar.subdetId == PixelSubdetector::PixelEndcap){
       unsigned int whichHalfCylinder(1);  //DetId does not know about halfCylinders in PXF
       if( tkTreeVar.posX < 0. ) whichHalfCylinder=2;
       
       // Phase 0
       // if( (rawId>=352394500 && rawId<352406032) || (rawId>=352460036 && rawId<352471568) || (rawId>=344005892 && rawId<344017424) || (rawId>=344071428 && rawId<344082960) )whichHalfCylinder=2;
       // Phase I       
       // if( (rawId>=352613380 && rawId<352655370) || (rawId>=352715780 && rawId<352782350) || (rawId>=352875524 && rawId<352917510) ||
       // (rawId>=352977924 && rawId<353044490) || (rawId>=353137668 && rawId<353179660) || (rawId>=353240068 && rawId<353306630) ||
       // (rawId>=344224772 && rawId<344266760) || (rawId>=344327172 && rawId<344393740) || (rawId>=344486916 && rawId<344528910) || 
       // (rawId>=344589316 && rawId<344655880) || (rawId>=344749060 && rawId<344791050) || (rawId>=344851460 && rawId<344918030) )whichHalfCylinder=2;
       
       tkTreeVar.layer  = tTopo->pxfDisk(*(&detId));
       tkTreeVar.side   =tTopo->pxfSide(*(&detId));
       tkTreeVar.half   = whichHalfCylinder;
       tkTreeVar.blade  = tTopo->pxfBlade(*(&detId));
       tkTreeVar.panel  = tTopo->pxfPanel(*(&detId));
       tkTreeVar.module = tTopo->pxfModule(*(&detId));
     }
     else if(tkTreeVar.subdetId == StripSubdetector::TIB){
       unsigned int whichHalfShell(1);  //DetId does not know about halfShells in TIB
       if( tkTreeVar.posY < 0. ) whichHalfShell=2;
       
       // if( (rawId>=369120484 && rawId<369120688) || (rawId>=369121540 && rawId<369121776) || (rawId>=369136932 && rawId<369137200) || (rawId>=369137988 && rawId<369138288) ||
       // (rawId>=369153396 && rawId<369153744) || (rawId>=369154436 && rawId<369154800) || (rawId>=369169844 && rawId<369170256) || (rawId>=369170900 && rawId<369171344) ||
	   // (rawId>=369124580 && rawId<369124784) || (rawId>=369125636 && rawId<369125872) || (rawId>=369141028 && rawId<369141296) || (rawId>=369142084 && rawId<369142384) ||
	   // (rawId>=369157492 && rawId<369157840) || (rawId>=369158532 && rawId<369158896) || (rawId>=369173940 && rawId<369174352) || (rawId>=369174996 && rawId<369175440) ) whichHalfShell=2;
       
       tkTreeVar.layer        = tTopo->tibLayer(*(&detId)); 
       tkTreeVar.side         = tTopo->tibStringInfo(*(&detId))[0];
       tkTreeVar.half         = whichHalfShell;
       tkTreeVar.rod          = tTopo->tibStringInfo(*(&detId))[2];
       tkTreeVar.outerInner   = tTopo->tibStringInfo(*(&detId))[1];
       tkTreeVar.module       = tTopo->tibModule(*(&detId));
       tkTreeVar.isDoubleSide = tTopo->tibIsDoubleSide(*(&detId));
       tkTreeVar.isRPhi       = tTopo->tibIsRPhi(*(&detId));
       tkTreeVar.isStereo       = tTopo->tibIsStereo(*(&detId));
     }
     else if(tkTreeVar.subdetId == StripSubdetector::TID){
       tkTreeVar.layer        = tTopo->tidWheel(*(&detId));
       tkTreeVar.side         = tTopo->tidSide(*(&detId));
       tkTreeVar.ring         = tTopo->tidRing(*(&detId));
       tkTreeVar.outerInner   = tTopo->tidModuleInfo(*(&detId))[0]; 
       tkTreeVar.module       = tTopo->tidModuleInfo(*(&detId))[1]; 
       tkTreeVar.isDoubleSide = tTopo->tidIsDoubleSide(*(&detId));
       tkTreeVar.isRPhi       = tTopo->tidIsRPhi(*(&detId));
       tkTreeVar.isStereo       = tTopo->tidIsStereo(*(&detId));
      } 
     else if(tkTreeVar.subdetId == StripSubdetector::TOB){
       tkTreeVar.layer        = tTopo->tobLayer(*(&detId)); 
       tkTreeVar.side         = tTopo->tobRodInfo(*(&detId))[0];
       tkTreeVar.rod          = tTopo->tobRodInfo(*(&detId))[1]; 
       tkTreeVar.module       = tTopo->tobModule(*(&detId));
       tkTreeVar.isDoubleSide = tTopo->tobIsDoubleSide(*(&detId));
       tkTreeVar.isRPhi       = tTopo->tobIsRPhi(*(&detId));
       tkTreeVar.isStereo       = tTopo->tobIsStereo(*(&detId));
     }
     else if(tkTreeVar.subdetId == StripSubdetector::TEC){
       tkTreeVar.layer        = tTopo->tecWheel(*(&detId));
       tkTreeVar.side         = tTopo->tecSide(*(&detId));
       tkTreeVar.ring         = tTopo->tecRing(*(&detId)); 
       tkTreeVar.petal        = tTopo->tecPetalInfo(*(&detId))[1]; 
       tkTreeVar.outerInner   = tTopo->tecPetalInfo(*(&detId))[0]; 
       tkTreeVar.module       = tTopo->tecModule(*(&detId));
       tkTreeVar.isDoubleSide = tTopo->tecIsDoubleSide(*(&detId));
       if ( tkTreeVar.ring == 1 || tkTreeVar.ring == 2 ||tkTreeVar.ring == 5 )
       { 
       tkTreeVar.isRPhi       = tTopo->tecIsRPhi(*(&detId));
       tkTreeVar.isStereo       = tTopo->tecIsStereo(*(&detId));
	   }
     }
     
     
     
     if(dynamic_cast<const StripGeomDetUnit*>(&geomDet)){  //is it a single physical module?
       const StripGeomDetUnit& StripgeomDetUnit = dynamic_cast<const StripGeomDetUnit&>(geomDet);
       if(tkTreeVar.subdetId==StripSubdetector::TIB || tkTreeVar.subdetId==StripSubdetector::TOB ||
          tkTreeVar.subdetId==StripSubdetector::TID || tkTreeVar.subdetId==StripSubdetector::TEC){
         const StripTopology& topol = dynamic_cast<const StripTopology&>(StripgeomDetUnit.specificTopology());
         tkTreeVar.nStrips = topol.nstrips();
       }
     }
     
     
     if(!createEntryForDoubleSidedModule_){if(tkTreeVar.isDoubleSide==1)continue;}  // do so only for individual modules and not also one entry for the combined doubleSided Module
     vTkTreeVar_.push_back(tkTreeVar);
   }
   
}


// ------------ method called once each job just before starting event loop  ------------
void 
TrackerTreeGenerator::beginJob()
{
}


// ------------ method called once each job just after ending the event loop  ------------
void 
TrackerTreeGenerator::endJob() {
   //~ UInt_t rawId(999), subdetId(999), layer(999), side(999), half(999), rod(999), ring(999), petal(999),
          //~ blade(999), panel(999), outerInner(999), module(999), rodAl(999), bladeAl(999), nStrips(999);
   UInt_t rawId(999), subdetId(999), layer(999), side(999), half(999), rod(999), ring(999), petal(999),
          blade(999), panel(999), outerInner(999), module(999), nStrips(999);
   Bool_t isDoubleSide(false), isRPhi(false), isStereo(false);
   Int_t uDirection(999), vDirection(999), wDirection(999);
   Float_t posR(999.F), posPhi(999.F), posEta(999.F), posX(999.F), posY(999.F), posZ(999.F);
   edm::Service<TFileService> fileService;
   TFileDirectory treeDir = fileService->mkdir("TrackerTree");
   TTree* trackerTree;
   trackerTree = treeDir.make<TTree>("TrackerTree","IDs of all modules (ideal geometry)");
   trackerTree->Branch("RawId", &rawId, "RawId/i");
   trackerTree->Branch("SubdetId", &subdetId, "SubdetId/i");
   trackerTree->Branch("Layer", &layer, "Layer/i");			      // Barrel: Layer, Forward: Disk
   trackerTree->Branch("Side", &side, "Side/i");			      // Rod/Ring in +z or -z
   trackerTree->Branch("Half", &half, "Half/i");                              // PXB: HalfBarrel, PXF: HalfCylinder, TIB: HalfShell
   trackerTree->Branch("Rod", &rod, "Rod/i"); 			              // Barrel (Ladder or String or Rod)
   trackerTree->Branch("Ring", &ring, "Ring/i");			      // Forward
   trackerTree->Branch("Petal", &petal, "Petal/i");			      // TEC
   trackerTree->Branch("Blade", &blade, "Blade/i");			      // PXF
   trackerTree->Branch("Panel", &panel, "Panel/i");			      // PXF
   trackerTree->Branch("OuterInner", &outerInner, "OuterInner/i");            // front/back String,Ring,Petal
   trackerTree->Branch("Module", &module, "Module/i");		              // Module ID
   trackerTree->Branch("NStrips", &nStrips, "NStrips/i");
   trackerTree->Branch("IsDoubleSide", &isDoubleSide, "IsDoubleSide/O");
   trackerTree->Branch("IsRPhi", &isRPhi, "IsRPhi/O");
   trackerTree->Branch("IsStereo", &isStereo, "IsStereo/O");
   trackerTree->Branch("UDirection", &uDirection, "UDirection/I");
   trackerTree->Branch("VDirection", &vDirection, "VDirection/I");
   trackerTree->Branch("WDirection", &wDirection, "WDirection/I");
   trackerTree->Branch("PosR", &posR, "PosR/F");
   trackerTree->Branch("PosPhi", &posPhi, "PosPhi/F");
   trackerTree->Branch("PosEta", &posEta, "PosEta/F");
   trackerTree->Branch("PosX", &posX, "PosX/F");
   trackerTree->Branch("PosY", &posY, "PosY/F");
   trackerTree->Branch("PosZ", &posZ, "PosZ/F");
   
   for(std::vector<TrackerTreeVariables>::const_iterator iTree = vTkTreeVar_.begin(); iTree != vTkTreeVar_.end(); ++iTree){
     rawId        = (*iTree).rawId;
     subdetId     = (*iTree).subdetId;
     layer        = (*iTree).layer;
     side         = (*iTree).side;
     half         = (*iTree).half;
     rod          = (*iTree).rod;
     ring         = (*iTree).ring;
     petal        = (*iTree).petal;
     blade        = (*iTree).blade;
     panel        = (*iTree).panel;
     outerInner   = (*iTree).outerInner;
     module       = (*iTree).module;
     nStrips      = (*iTree).nStrips;
     isDoubleSide = (*iTree).isDoubleSide;
     isRPhi       = (*iTree).isRPhi;
     isStereo       = (*iTree).isStereo;
     uDirection   = (*iTree).uDirection;
     vDirection   = (*iTree).vDirection;
     wDirection   = (*iTree).wDirection;
     posR         = (*iTree).posR;
     posPhi       = (*iTree).posPhi;
     posEta       = (*iTree).posEta;
     posX         = (*iTree).posX;
     posY         = (*iTree).posY;
     posZ         = (*iTree).posZ;
     
     trackerTree->Fill();
   }
   edm::LogInfo("TrackerTreeGenerator") << "TrackerTree contains "<< vTkTreeVar_.size() <<" entries overall";
}


//define this as a plug-in
DEFINE_FWK_MODULE(TrackerTreeGenerator);
