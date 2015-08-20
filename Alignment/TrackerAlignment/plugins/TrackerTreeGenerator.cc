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
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/Math/interface/deltaPhi.h"

//#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
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
   
   edm::ESHandle<PTrackerParameters> ptp;
   iSetup.get<PTrackerParametersRcd>().get(ptp);
   TrackerGeomBuilderFromGeometricDet trackerBuilder;
   TrackerGeometry* tkGeom = trackerBuilder.build(&(*geometricDet), *ptp);
 
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
     
     if(tkTreeVar.subdetId == PixelSubdetector::PixelBarrel){
       PXBDetId pxbId(rawId);
       unsigned int whichHalfBarrel(1), ladderAl(0);  //DetId does not know about halfBarrels is PXB ...
       if( (rawId>=302056964 && rawId<302059300) || (rawId>=302123268 && rawId<302127140) || (rawId>=302189572 && rawId<302194980) )whichHalfBarrel=2;
       tkTreeVar.layer  = pxbId.layer();
       tkTreeVar.half   = whichHalfBarrel;
       tkTreeVar.rod    = pxbId.ladder();     // ... so, ladder is not per halfBarrel-Layer, but per barrel-layer!
       tkTreeVar.module = pxbId.module();
       if(tkTreeVar.layer==1){
         if(tkTreeVar.half==2)ladderAl = tkTreeVar.rod -5;
         else if(tkTreeVar.rod>15)ladderAl = tkTreeVar.rod -10;
         else ladderAl = tkTreeVar.rod;
       }else if(tkTreeVar.layer==2){
         if(tkTreeVar.half==2)ladderAl = tkTreeVar.rod -8;
         else if(tkTreeVar.rod>24)ladderAl = tkTreeVar.rod -16;
         else ladderAl = tkTreeVar.rod;
       }else if(tkTreeVar.layer==3){
         if(tkTreeVar.half==2)ladderAl = tkTreeVar.rod -11;
         else if(tkTreeVar.rod>33)ladderAl = tkTreeVar.rod -22;
         else ladderAl = tkTreeVar.rod;
       }
       tkTreeVar.rodAl = ladderAl;}
     else if(tkTreeVar.subdetId == PixelSubdetector::PixelEndcap){
       PXFDetId pxfId(rawId);
       unsigned int whichHalfCylinder(1), bladeAl(0);  //DetId does not kmow about halfCylinders in PXF
       if( (rawId>=352394500 && rawId<352406032) || (rawId>=352460036 && rawId<352471568) || (rawId>=344005892 && rawId<344017424) || (rawId>=344071428 && rawId<344082960) )whichHalfCylinder=2;
       tkTreeVar.layer  = pxfId.disk();
       tkTreeVar.side   = pxfId.side();
       tkTreeVar.half   = whichHalfCylinder;
       tkTreeVar.blade  = pxfId.blade();
       tkTreeVar.panel  = pxfId.panel();
       tkTreeVar.module = pxfId.module();
       if(tkTreeVar.half==2)bladeAl = tkTreeVar.blade -6;
       else if(tkTreeVar.blade>18)bladeAl = tkTreeVar.blade -12;
       else bladeAl = tkTreeVar.blade;
       tkTreeVar.bladeAl = bladeAl;}
     else if(tkTreeVar.subdetId == StripSubdetector::TIB){
       TIBDetId tibId(rawId);
       unsigned int whichHalfShell(1), stringAl(0);  //DetId does not kmow about halfShells in TIB
       if( (rawId>=369120484 && rawId<369120688) || (rawId>=369121540 && rawId<369121776) || (rawId>=369136932 && rawId<369137200) || (rawId>=369137988 && rawId<369138288) ||
           (rawId>=369153396 && rawId<369153744) || (rawId>=369154436 && rawId<369154800) || (rawId>=369169844 && rawId<369170256) || (rawId>=369170900 && rawId<369171344) ||
	   (rawId>=369124580 && rawId<369124784) || (rawId>=369125636 && rawId<369125872) || (rawId>=369141028 && rawId<369141296) || (rawId>=369142084 && rawId<369142384) ||
	   (rawId>=369157492 && rawId<369157840) || (rawId>=369158532 && rawId<369158896) || (rawId>=369173940 && rawId<369174352) || (rawId>=369174996 && rawId<369175440) ) whichHalfShell=2;
       tkTreeVar.layer        = tibId.layer();
       tkTreeVar.side         = tibId.string()[0];
       tkTreeVar.half         = whichHalfShell;
       tkTreeVar.rod          = tibId.string()[2];
       tkTreeVar.outerInner   = tibId.string()[1];
       tkTreeVar.module       = tibId.module();
       tkTreeVar.isDoubleSide = tibId.isDoubleSide();
       tkTreeVar.isRPhi       = tibId.isRPhi();
       if(tkTreeVar.half==2){
         if(tkTreeVar.layer==1){
           if(tkTreeVar.outerInner==1)stringAl = tkTreeVar.rod -13;
	   else if(tkTreeVar.outerInner==2)stringAl = tkTreeVar.rod -15;
         }
         if(tkTreeVar.layer==2){
           if(tkTreeVar.outerInner==1)stringAl = tkTreeVar.rod -17;
	   else if(tkTreeVar.outerInner==2)stringAl = tkTreeVar.rod -19;
         }
         if(tkTreeVar.layer==3){
           if(tkTreeVar.outerInner==1)stringAl = tkTreeVar.rod -22;
	   else if(tkTreeVar.outerInner==2)stringAl = tkTreeVar.rod -23;
         }
         if(tkTreeVar.layer==4){
           if(tkTreeVar.outerInner==1)stringAl = tkTreeVar.rod -26;
	   else if(tkTreeVar.outerInner==2)stringAl = tkTreeVar.rod -28;
         }
       }
       else stringAl = tkTreeVar.rod;
       tkTreeVar.rodAl = stringAl;}
     else if(tkTreeVar.subdetId == StripSubdetector::TID){
       TIDDetId tidId(rawId);
       tkTreeVar.layer        = tidId.wheel();
       tkTreeVar.side         = tidId.side();
       tkTreeVar.ring         = tidId.ring();
       tkTreeVar.outerInner   = tidId.module()[0];
       tkTreeVar.module       = tidId.module()[1];
       tkTreeVar.isDoubleSide = tidId.isDoubleSide();
       tkTreeVar.isRPhi       = tidId.isRPhi();}
     else if(tkTreeVar.subdetId == StripSubdetector::TOB){
       TOBDetId tobId(rawId);
       tkTreeVar.layer        = tobId.layer();
       tkTreeVar.side         = tobId.rod()[0];
       tkTreeVar.rod          = tobId.rod()[1];
       tkTreeVar.module       = tobId.module();
       tkTreeVar.isDoubleSide = tobId.isDoubleSide();
       tkTreeVar.isRPhi       = tobId.isRPhi();}
     else if(tkTreeVar.subdetId == StripSubdetector::TEC){
       TECDetId tecId(rawId);
       tkTreeVar.layer        = tecId.wheel();
       tkTreeVar.side         = tecId.side();
       tkTreeVar.ring         = tecId.ring();
       tkTreeVar.petal        = tecId.petal()[1];
       tkTreeVar.outerInner   = tecId.petal()[0];
       tkTreeVar.module       = tecId.module();
       tkTreeVar.isDoubleSide = tecId.isDoubleSide();
       tkTreeVar.isRPhi       = tecId.isRPhi();}
     
     
     LocalPoint lPModule(0.,0.,0.), lUDirection(1.,0.,0.), lVDirection(0.,1.,0.), lWDirection(0.,0.,1.);
     GlobalPoint gPModule    = surface.toGlobal(lPModule),
                 gUDirection = surface.toGlobal(lUDirection),
                 gVDirection = surface.toGlobal(lVDirection),
	         gWDirection = surface.toGlobal(lWDirection);
     double dR(999.), dPhi(999.), dZ(999.);
     if(tkTreeVar.subdetId==PixelSubdetector::PixelBarrel || tkTreeVar.subdetId==StripSubdetector::TIB
                                                          || tkTreeVar.subdetId==StripSubdetector::TOB){
       dR   = gWDirection.perp() - gPModule.perp();
       dPhi = deltaPhi(gUDirection.phi(),gPModule.phi());
       dZ   = gVDirection.z() - gPModule.z();
       tkTreeVar.uDirection = dPhi>0. ? 1 : -1;
       tkTreeVar.vDirection = dZ>0.   ? 1 : -1;
       tkTreeVar.wDirection = dR>0.   ? 1 : -1;
     }else if(tkTreeVar.subdetId==PixelSubdetector::PixelEndcap){
       dR   = gUDirection.perp() - gPModule.perp();
       dPhi = deltaPhi(gVDirection.phi(),gPModule.phi());
       dZ   = gWDirection.z() - gPModule.z();
       tkTreeVar.uDirection = dR>0.   ? 1 : -1;
       tkTreeVar.vDirection = dPhi>0. ? 1 : -1;
       tkTreeVar.wDirection = dZ>0.   ? 1 : -1;
     }else if(tkTreeVar.subdetId==StripSubdetector::TID || tkTreeVar.subdetId==StripSubdetector::TEC){
       dR = gVDirection.perp() - gPModule.perp();
       dPhi = deltaPhi(gUDirection.phi(),gPModule.phi());
       dZ = gWDirection.z() - gPModule.z();
       tkTreeVar.uDirection = dPhi>0. ? 1 : -1;
       tkTreeVar.vDirection = dR>0.   ? 1 : -1;
       tkTreeVar.wDirection = dZ>0.   ? 1 : -1;
     }
     tkTreeVar.posR         = gPModule.perp();
     tkTreeVar.posPhi       = gPModule.phi();     // = gPModule.phi().degrees();
     tkTreeVar.posEta       = gPModule.eta();
     tkTreeVar.posX         = gPModule.x();
     tkTreeVar.posY         = gPModule.y();
     tkTreeVar.posZ         = gPModule.z();
     
     
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
   UInt_t rawId(999), subdetId(999), layer(999), side(999), half(999), rod(999), ring(999), petal(999),
          blade(999), panel(999), outerInner(999), module(999), rodAl(999), bladeAl(999), nStrips(999);
   Bool_t isDoubleSide(false), isRPhi(false);
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
   trackerTree->Branch("RodAl", &rodAl, "RodAl/i");                           // Different for AlignmentHierarchy from TrackerHierarchy (TPB, TIB)
   trackerTree->Branch("BladeAl", &bladeAl, "BladeAl/i");                     // Different for AlignmentHierarchy from TrackerHierarchy (TPF)
   trackerTree->Branch("NStrips", &nStrips, "NStrips/i");
   trackerTree->Branch("IsDoubleSide", &isDoubleSide, "IsDoubleSide/O");
   trackerTree->Branch("IsRPhi", &isRPhi, "IsRPhi/O");
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
     rodAl        = (*iTree).rodAl;
     bladeAl      = (*iTree).bladeAl;
     nStrips      = (*iTree).nStrips;
     isDoubleSide = (*iTree).isDoubleSide;
     isRPhi       = (*iTree).isRPhi;
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
