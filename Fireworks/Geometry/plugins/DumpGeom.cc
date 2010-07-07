// -*- C++ -*-
//
// Package:    DumpGeom
// Class:      DumpGeom
// 
/**\class DumpGeom DumpGeom.cc Reve/DumpGeom/src/DumpGeom.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris D Jones
//         Created:  Wed Sep 26 08:27:23 EDT 2007
// $Id: DumpGeom.cc,v 1.4 2010/07/02 14:53:41 mccauley Exp $
//
//

// system include files
#include <memory>
#include <iostream>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "Fireworks/Geometry/interface/DisplayGeomRecord.h"

#include "TGeoManager.h"
#include "TCanvas.h"
#include "TGeoPcon.h"
#include "TGeoPgon.h"
#include "TGeoCone.h"
#include "TGeoBoolNode.h"
#include "TGeoTube.h"
#include "TGeoCompositeShape.h"
#include "TGeoArb8.h"
#include "TGeoTrd2.h"
#include "TGeoMatrix.h"
#include "TFile.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "Math/GenVector/RotationX.h"

///////////////////////////////////////////////////////////
// Muons

#include <Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/DTGeometry/interface/DTChamber.h>
#include <Geometry/DTGeometry/interface/DTLayer.h>

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>

#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/MuonNumbering/interface/MuonDDDNumbering.h>
#include <Geometry/MuonNumbering/interface/MuonBaseNumber.h>
#include <Geometry/MuonNumbering/interface/DTNumberingScheme.h>
#include <Geometry/MuonNumbering/interface/CSCNumberingScheme.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/Records/interface/MuonNumberingRecord.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/MuonNumbering/interface/RPCNumberingScheme.h>
#include <CondFormats/GeometryObjects/interface/RecoIdealGeometry.h>
#include <CondFormats/GeometryObjects/interface/CSCRecoDigiParameters.h>
#include <Geometry/CSCGeometryBuilder/src/CSCGeometryParsFromDD.h>

#include <Geometry/TrackerNumberingBuilder/interface/GeometricDet.h>

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/EcalCommonData/interface/EcalBarrelNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalEndcapNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalPreshowerNumberingScheme.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include "TTree.h"
#include "TError.h"

//
// class decleration
//

class DumpGeom : public edm::EDAnalyzer
{
   public:
      explicit DumpGeom(const edm::ParameterSet&);
      ~DumpGeom();

  template <class T> friend class CaloGeometryLoader;//<EcalBarralGeometry>;

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      void mapDTGeometry(const DDCompactView& cview,
			 const MuonDDDConstants& muonConstants,
			 const DTGeometry& dtGeom);
      void mapCSCGeometry(const DDCompactView& cview,
			  const MuonDDDConstants& muonConstants);
      void mapTrackerGeometry(const DDCompactView& cview,
			      const GeometricDet& gd);
      void mapEcalGeometry(const DDCompactView& cview,
                           const CaloGeometry& cg);
      void mapRPCGeometry(const DDCompactView& cview,
                          const MuonDDDConstants& muonConstants);

      // ----------member data ---------------------------
      struct Info{
	std::string name;
	Float_t points[24]; // x1,y1,z1...x8,y8,z8
	Info(const std::string& iname):
	  name(iname){
	  init();
	}
	Info(){
	  init();
	}
	void init(){
	  for(unsigned int i=0; i<24; ++i) points[i]=0;
	}
	void fillPoints(std::vector<GlobalPoint>::const_iterator begin, std::vector<GlobalPoint>::const_iterator end)
	{
	  unsigned int index(0);
	  for(std::vector<GlobalPoint>::const_iterator i = begin; i!=end; ++i){
	    assert(index<8);
	    points[index*3] = i->x();
	    points[index*3+1] = i->y();
	    points[index*3+2] = i->z();
	    ++index;
	  }
	}
      };

      std::map<unsigned int, Info>         idToName_;
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
DumpGeom::DumpGeom(const edm::ParameterSet&)
{
   // now do what ever initialization is needed
}


DumpGeom::~DumpGeom()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


void DumpGeom::mapDTGeometry(const DDCompactView& cview,
			     const MuonDDDConstants& muonConstants,
			     const DTGeometry& dtGeom)
{
   // filter out everythin but DT muon geometry
   std::string attribute = "MuStructure"; 
   std::string value     = "MuonBarrelDT";
   DDValue val(attribute, value, 0.0);

   // Asking only for the Muon DTs
   DDSpecificsFilter filter;
   filter.setCriteria(val,  // name & value of a variable 
		      DDSpecificsFilter::matches,
		      DDSpecificsFilter::AND, 
		      true, // compare strings otherwise doubles
		      true  // use merged-specifics or simple-specifics
		      );
   DDFilteredView fview(cview);
   fview.addFilter(filter);
   
   bool doChamber = fview.firstChild();

   // Loop on chambers
   while (doChamber){
      std::stringstream s;
      s << "/cms:World_1";
      DDGeoHistory::const_iterator ancestor = fview.geoHistory().begin();
      ++ancestor; // skip the first ancestor
      for ( ; ancestor != fview.geoHistory().end(); ++ ancestor )
	s << "/" << ancestor->logicalPart().name() << "_" << ancestor->copyno();
      
      std::string name = s.str();
      
      MuonDDDNumbering mdddnum (muonConstants);
      DTNumberingScheme dtnum (muonConstants);
      
      // this doesn't work
      // unsigned int rawid = dtnum.baseNumberToUnitNumber( mdddnum.geoHistoryToBaseNumber( fview.geoHistory() ) );

      // FIX ME - hack to pretend that we have more layers than we really have
      MuonBaseNumber hackedMuonBaseNumber( mdddnum.geoHistoryToBaseNumber( fview.geoHistory() ) );
      hackedMuonBaseNumber.addBase(4,0,0);
      hackedMuonBaseNumber.addBase(5,0,0);
      hackedMuonBaseNumber.addBase(6,0,0);
      DTChamberId cid(dtnum.baseNumberToUnitNumber( hackedMuonBaseNumber ));
      unsigned int rawid = cid.rawId();

      // this works fine if you have access
      // unsigned int rawid = dtnum.getDetId( mdddnum.geoHistoryToBaseNumber( fview.geoHistory() ) );
      //      std::cout << "DT chamber id: " << rawid << " \tname: " << name << std::endl;
      
      idToName_[rawid] = Info(name);
      
      doChamber = fview.nextSibling(); // go to next chamber
   }

   // Fill in DT super layer parameters
   for(std::vector<DTSuperLayer*>::const_iterator det = dtGeom.superLayers().begin(); 
       det != dtGeom.superLayers().end(); ++det)
   {
     unsigned int rawId = (*det)->id().rawId();
     const BoundPlane& surf = (*det)->surface();
     // bounds W/H/L:
     idToName_[rawId].points[0] = surf.bounds().width();
     idToName_[rawId].points[1] = surf.bounds().thickness();
     idToName_[rawId].points[2] = surf.bounds().length();
   }

   // Fill in DT layer parameters
   for(std::vector<DTLayer*>::const_iterator det = dtGeom.layers().begin(); 
       det != dtGeom.layers().end(); ++det)
   {
     unsigned int rawId = (*det)->id().rawId();
     const DTTopology& topo = (*det)->specificTopology();
     const BoundPlane& surf = (*det)->surface();
     // Topology W/H/L:
     idToName_[rawId].points[0] = topo.cellWidth();
     idToName_[rawId].points[1] = topo.cellHeight();
     idToName_[rawId].points[2] = topo.cellLenght();
     idToName_[rawId].points[3] = topo.firstChannel();
     idToName_[rawId].points[4] = topo.lastChannel();
     idToName_[rawId].points[5] = topo.channels();

     // bounds W/H/L:
     idToName_[rawId].points[6] = surf.bounds().width();
     idToName_[rawId].points[7] = surf.bounds().thickness();
     idToName_[rawId].points[8] = surf.bounds().length();

     for( int i = 0; i < 8; ++i)
       std::cout << idToName_[rawId].points[i] << ", ";
     std::cout << std::endl;
   }
}

/** 
 ** By Michael Case
 ** method mapCSCGeometry(...)
 ** date: 01-25-2008
 ** Description:
 ** Assign layer det id's to a DD "path" or "geo History".
 ** date: 03-22-2010, MEC
 **      Added the layers last Nov.  Fixed a bug just now.
 **/
void DumpGeom::mapCSCGeometry(const DDCompactView& cview,
			     const MuonDDDConstants& muonConstants) {

  // use of new code factoring of the Builder to be used by the Reco DB.
  RecoIdealGeometry rig;
  // not sure I need this... but DO need it to build the actual geometry.
  CSCRecoDigiParameters rdp;
  
  // simple class just really a method to get the parameters... but I want this method
  // available to classes other than CSCGeometryBuilderFromDDD so... simple class...
  CSCGeometryParsFromDD cscp;
  if ( ! cscp.build(&cview, muonConstants, rig, rdp) ) {
    throw cms::Exception("CSCGeometryBuilderFromDDD", "Failed to build the necessary objects from the DDD");
  }
 
  const std::vector<DetId>& did = rig.detIds();
  std::vector<double> trans, rot;
  //  std::cout << did.size() << " Number of CSC Chambers" << std::endl;

  std::string myName="DumpCSCGeom";
  std::string attribute = "MuStructure"; 
  std::string value     = "MuonEndcapCSC";
  DDValue val(attribute, value, 0.0);
  
  // Asking only for the Muon CSCs
  DDSpecificsFilter filter;
  filter.setCriteria(val,  // name & value of a variable 
		     DDSpecificsFilter::equals,
		     DDSpecificsFilter::AND, 
		     true, // compare strings otherwise doubles
		     true  // use merged-specifics or simple-specifics
		     );
  DDFilteredView fview(cview);
  fview.addFilter(filter);
  //  std::cout << "****************about to skip firstChild() ONCE" << std::endl;   
  bool doSubDets = fview.firstChild();

  //  Loop on chambers
  //  Since we have the RIG (RecoIdealGeometry) detIds, we loop over this filter
  //  then look up the detID.
  while (doSubDets){

    /// Naming block
    // this will still work w/ CSC's but only goes down to the Chamber level
    std::stringstream s;
    s << "/cms:World_1";
    DDGeoHistory::const_iterator ancestor = fview.geoHistory().begin();
    ++ancestor; // skip the first ancestor
    for ( ; ancestor != fview.geoHistory().end(); ++ ancestor )
      s << "/" << ancestor->logicalPart().name() << "_" << ancestor->copyno();
      
    std::string name = s.str();
      
    MuonDDDNumbering mdn(muonConstants);
    MuonBaseNumber mbn = mdn.geoHistoryToBaseNumber(fview.geoHistory());
    CSCNumberingScheme mens(muonConstants);
    int id = mens.baseNumberToUnitNumber( mbn );
    CSCDetId chamberId(id);

    DetId searchId (chamberId);
    std::vector<DetId>::const_iterator findIt = std::find(did.begin(), did.end(), searchId);

    // The above gives you the CHAMBER DetID but not the LAYER.
    // For CSCs the layers are built from specpars of the chamber.
    // this code is from CSCGeometryBuilder package in Geometry subsystem.
    int jend   = chamberId.endcap();
    int jstat  = chamberId.station();
    int jring  = chamberId.ring();
    int jch    = chamberId.chamber();

    int localZwrtGlobalZ = +1;
    if ( (jend==1 && jstat<3 ) || ( jend==2 && jstat>2 ) ) localZwrtGlobalZ = -1;
    int globalZ = +1;
    if ( jend == 2 ) globalZ = -1;

    idToName_[chamberId.rawId()] = Info(name);
    //    std::cout << "CSC chamber detID: " << chamberId<< " "<< chamberId.rawId() << " \tname: " << name << std::endl;
    
    for ( short j = 1; j <= 6; ++j ) {
      std::string layerName = name;
      CSCDetId layerId = CSCDetId( jend, jstat, jring, jch, j );
      
      DetId searchId2 (layerId);
      std::vector<DetId>::const_iterator findIt = std::find(did.begin(), did.end(), searchId2);
      
      unsigned int rawid = layerId.rawId();

      //  Go down to  ME11_Layer.
      std::string prefName = (fview.logicalPart().name().name()).substr(0,4);
      //ME21FR4Body 1
      //      10 mf:ME11PolycarbPanel 1 Trapezoid
      layerName += "/mf:" + prefName + "FR4Body_1/mf:" + prefName + "PolycarbPanel_1/mf:" + prefName + "Layer_"; //"_ActiveGasVol_";
      std::ostringstream ostr;
      ostr << j;
      layerName += ostr.str();
      idToName_[rawid] = layerName;
      //      std::cout << "CSC Layer   detID: " << layerId << " " << rawid << " \tname: " << layerName << std::endl;

    } // layer construction within chamber
     
     //  If it's ME11 you need to have two detId's per chamber. This is how to construct the detId
     //  copied from the CSCGeometryBuilder code.
     if ( jstat==1 && jring==1 ) {
       CSCDetId detid1a = CSCDetId( jend, 1, 4, jch, 0 );
       // the chamber "name" is the same for both detId's, I believe.
       //       std::cout << "CSC Chamber detID: " <<detid1a<<" "<< detid1a.rawId() << " \tname: " << name << std::endl;
       idToName_[detid1a.rawId()] = Info(name);
       for ( short j = 1; j <= 6; ++j ) {
	 std::string layerName = name;
	 CSCDetId layerId = CSCDetId( jend, 1, 4, jch, j );
	 DetId searchId2 (layerId);
	 std::vector<DetId>::const_iterator findIt = std::find(did.begin(), did.end(), searchId2);
	 //      if ( findIt == did.end() ) std::cout << "DID NOT find layer DetId in RecoIdealGeometry object." << std::endl;
	 //      else std::cout << "Found layer DetID in RecoIdealGeometry object" << std::endl;
	 unsigned int rawid = layerId.rawId();
	 std::string prefName = (fview.logicalPart().name().name()).substr(0,4);
	 layerName += "/mf:" + prefName + "FR4Body_1/mf:" + prefName + "PolycarbPanel_1/mf:" + prefName + "Layer_"; //"_ActiveGasVol_";
	 std::ostringstream ostr;
	 ostr << j;
	 layerName += ostr.str();
	 idToName_[rawid] = layerName;
	 //	 std::cout << "CSC Layer   detID: " << layerId << " " << rawid << " \tname: " << layerName << std::endl;
       } // layer construction within chamber
     }
    doSubDets = fview.nextSibling(); // go to next chamber
  }
}

/**
 ** By Michael Case
 ** method mapTrackerGeometry(...)
 ** date: 01-30-2008
 ** Description:
 **   Map tracker DetId to DD path (nav_type and GeometricDet are easiest way to get it).
 **   Note: later, may need pset bool "fromDD" because tracker now has capability to retrieve
 **         persistent GeometricDet from Conditions DB.
 **/
void DumpGeom::mapTrackerGeometry(const DDCompactView& cview,
				  const GeometricDet& rDD) {
  const GeometricDet::ConstGeometricDetContainer& cgdc = rDD.deepComponents();
  GeometricDet::ConstGeometricDetContainer::const_iterator git = cgdc.begin();
  GeometricDet::ConstGeometricDetContainer::const_iterator egit = cgdc.end();
  DDExpandedView expv(cview);
  int id;
  for ( ; git != egit; ++git ) {
    expv.goTo( (*git)->navpos() );
    //    expv.goTo( (*git)->navType() );

    std::stringstream s;
    s << "/cms:World_1";
    DDGeoHistory::const_iterator ancestor = expv.geoHistory().begin();
    ++ancestor; // skip the first ancestor
    for ( ; ancestor != expv.geoHistory().end(); ++ ancestor )
      s << "/" << ancestor->logicalPart().name() << "_" << ancestor->copyno();
      
    std::string name = s.str();
    id = int((*git)->geographicalID());
    //    std::cout << "Tracker id: " << id << " \tname: " << name << std::endl;
    idToName_[id] = Info(name);
  }

}

/**
 ** By Michael Case
 ** method mapEcalGeometry(...)
 ** date: 02-07-2008
 ** Description:
 **   Map Ecal DetId to DD path 
 **   The code for CaloGeometry, EcalBarrelAlgo, EcalEndcapAlgo were all modified in
 **   The 169 series.  The correction WILL be different for 18X.  The files should
 **   be located on /afs/cern.ch/user/c/case/public/fwevtstuff/.
 **/
void DumpGeom::mapEcalGeometry(const DDCompactView& cview,
			       const CaloGeometry& cg) {
  {
    const CaloSubdetectorGeometry* geom=cg.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    
    // This code comes from CaloGeometryLoader and must be updated when CaloGeometryLoader changes.
    // it is cut-pasted (logic wise).
    DDSpecificsFilter filter;
    filter.setCriteria( DDValue( "SensitiveDetector",
				 "EcalSensitiveDetector",
				 0                        ),
			DDSpecificsFilter::equals,
			DDSpecificsFilter::AND,
			true,
			true                               ) ;
    
    filter.setCriteria( DDValue( "ReadOutName",
				 (dynamic_cast<const EcalBarrelGeometry*>(geom))->hitString(),
				 0                  ),
			DDSpecificsFilter::equals,
			DDSpecificsFilter::AND,
			true,
			true                       ) ;
    size_t tid;
    DDFilteredView fview(cview);
    fview.addFilter(filter);
    bool doSubDets = fview.firstChild();
    EcalBarrelNumberingScheme scheme;
    while (doSubDets) {
      const DDGeoHistory& parents ( fview.geoHistory() ) ;
      const DDGeoHistory::size_type psize ( parents.size() ) ;
      EcalBaseNumber baseNumber ;
      baseNumber.setSize( psize ) ;

    for( unsigned int i=1 ; i<=psize ; ++i )
      {
	baseNumber.addLevel( parents[psize-i].logicalPart().name().name(),
			     parents[psize-i].copyno() ) ;
      }

     tid = scheme.getUnitID( baseNumber );
     std::stringstream s;
     s << "/cms:World_1";
     DDGeoHistory::const_iterator ancestor = fview.geoHistory().begin();
     ++ancestor; // skip the first ancestor
     for ( ; ancestor != fview.geoHistory().end(); ++ ancestor )
       s << "/" << ancestor->logicalPart().name() << "_" << ancestor->copyno();
     
     std::string name = s.str();
     idToName_[tid] = Info(name);
     doSubDets = fview.nextSibling(); // go to next
    }
  }

  //  build(*pG,DetId::Ecal,EcalEndcap,*pDD);
  {
  const CaloSubdetectorGeometry* geom=cg.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    // This code comes from CaloGeometryLoader and must be updated when CaloGeometryLoader changes.
    // it is cut-pasted (logic wise).
    DDSpecificsFilter filter;
    filter.setCriteria( DDValue( "SensitiveDetector",
				 "EcalSensitiveDetector",
				 0                        ),
			DDSpecificsFilter::equals,
			DDSpecificsFilter::AND,
			true,
			true                               ) ;
    
    filter.setCriteria( DDValue( "ReadOutName",
				 (dynamic_cast<const EcalPreshowerGeometry*>(geom))->hitString(),
				 0                  ),
			DDSpecificsFilter::equals,
			DDSpecificsFilter::AND,
			true,
			true                       ) ;
    size_t tid;
    DDFilteredView fview(cview);
    fview.addFilter(filter);
    bool doSubDets = fview.firstChild();
    EcalPreshowerNumberingScheme scheme;
    while (doSubDets) {
      const DDGeoHistory& parents ( fview.geoHistory() ) ;
      const DDGeoHistory::size_type psize ( parents.size() ) ;
      EcalBaseNumber baseNumber ;
      baseNumber.setSize( psize ) ;

    for( unsigned int i=1 ; i<=psize ; ++i )
      {
	baseNumber.addLevel( parents[psize-i].logicalPart().name().name(),
			     parents[psize-i].copyno() ) ;
      }

     tid = scheme.getUnitID( baseNumber );  
     std::stringstream s;
     s << "/cms:World_1";
     DDGeoHistory::const_iterator ancestor = fview.geoHistory().begin();
     ++ancestor; // skip the first ancestor
     for ( ; ancestor != fview.geoHistory().end(); ++ ancestor )
       s << "/" << ancestor->logicalPart().name() << "_" << ancestor->copyno();
     
     std::string name = s.str();
     idToName_[tid] = Info(name);
     doSubDets = fview.nextSibling(); // go to next
    }

  }

  // preshower
  {
    const CaloSubdetectorGeometry* geom=cg.getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
    // This code comes from CaloGeometryLoader and must be updated when CaloGeometryLoader changes.
    // it is cut-pasted (logic wise).
    DDSpecificsFilter filter;
    filter.setCriteria( DDValue( "SensitiveDetector",
				 "EcalSensitiveDetector",
				 0                        ),
			DDSpecificsFilter::equals,
			DDSpecificsFilter::AND,
			true,
			true                               ) ;
    
    filter.setCriteria( DDValue( "ReadOutName",
				 (dynamic_cast<const EcalEndcapGeometry*>(geom))->hitString(),
				 0                  ),
			DDSpecificsFilter::equals,
			DDSpecificsFilter::AND,
			true,
			true                       ) ;
    size_t tid;
    DDFilteredView fview(cview);
    fview.addFilter(filter);
    bool doSubDets = fview.firstChild();
    EcalEndcapNumberingScheme scheme;
    while (doSubDets) {
      const DDGeoHistory& parents ( fview.geoHistory() ) ;
      const DDGeoHistory::size_type psize ( parents.size() ) ;
      EcalBaseNumber baseNumber ;
      baseNumber.setSize( psize ) ;

    for( unsigned int i=1 ; i<=psize ; ++i )
      {
	baseNumber.addLevel( parents[psize-i].logicalPart().name().name(),
			     parents[psize-i].copyno() ) ;
      }

     tid = scheme.getUnitID( baseNumber );  
     std::stringstream s;
     s << "/cms:World_1";
     DDGeoHistory::const_iterator ancestor = fview.geoHistory().begin();
     ++ancestor; // skip the first ancestor
     for ( ; ancestor != fview.geoHistory().end(); ++ ancestor )
       s << "/" << ancestor->logicalPart().name() << "_" << ancestor->copyno();
     
     std::string name = s.str();
     idToName_[tid] = Info(name);
     doSubDets = fview.nextSibling(); // go to next
    }

  }
  // HcalBarrel
  {
    std::vector<DetId> ids = cg.getValidDetIds(DetId::Hcal, HcalBarrel); //HB
    for(std::vector<DetId>::const_iterator id = ids.begin(), idEnd = ids.end(); id != idEnd; ++id){
      const CaloCellGeometry::CornersVec& cor (cg.getSubdetectorGeometry(*id)->getGeometry(*id)->getCorners());
      idToName_[id->rawId()].fillPoints(cor.begin(),cor.end());
    }
  }
  // HcalEndcap
  {
    std::vector<DetId> ids = cg.getValidDetIds(DetId::Hcal, HcalEndcap); //HE
    for(std::vector<DetId>::const_iterator id = ids.begin(), idEnd = ids.end(); id != idEnd; ++id){
      const CaloCellGeometry::CornersVec& cor (cg.getSubdetectorGeometry(*id)->getGeometry(*id)->getCorners());
      idToName_[id->rawId()].fillPoints(cor.begin(),cor.end());
    }
  }
  // HcalOuter
  {
    std::vector<DetId> ids = cg.getValidDetIds(DetId::Hcal, HcalOuter); //HO
    for(std::vector<DetId>::const_iterator id = ids.begin(), idEnd = ids.end(); id != idEnd; ++id){
      const CaloCellGeometry::CornersVec& cor (cg.getSubdetectorGeometry(*id)->getGeometry(*id)->getCorners());
      idToName_[id->rawId()].fillPoints(cor.begin(),cor.end());
    }
  }
  // HcalForward
  {
    std::vector<DetId> ids = cg.getValidDetIds(DetId::Hcal, HcalForward); //HF
    for(std::vector<DetId>::const_iterator id = ids.begin(), idEnd = ids.end(); id != idEnd; ++id){
      const CaloCellGeometry::CornersVec& cor (cg.getSubdetectorGeometry(*id)->getGeometry(*id)->getCorners());
      idToName_[id->rawId()].fillPoints(cor.begin(),cor.end());
    }
  }

  // Fill reco geometry
  {
    std::vector<DetId> ids = cg.getValidDetIds(DetId::Ecal, EcalBarrel);//EB
    for(std::vector<DetId>::const_iterator id = ids.begin(), idEnd = ids.end(); id != idEnd; ++id){
      const CaloCellGeometry::CornersVec& cor (cg.getSubdetectorGeometry(*id)->getGeometry(*id)->getCorners()) ;
      idToName_[id->rawId()].fillPoints(cor.begin(),cor.end());
    }
  }
  {
    std::vector<DetId> ids = cg.getValidDetIds(DetId::Ecal, EcalEndcap);//EE
    for(std::vector<DetId>::const_iterator id = ids.begin(), idEnd = ids.end(); id != idEnd; ++id){
      const CaloCellGeometry::CornersVec& cor (cg.getSubdetectorGeometry(*id)->getGeometry(*id)->getCorners()) ;
      idToName_[id->rawId()].fillPoints(cor.begin(),cor.end());
    }
  }
  {
    std::vector<DetId> ids = cg.getValidDetIds(DetId::Ecal, EcalPreshower);//ES
    for(std::vector<DetId>::const_iterator id = ids.begin(), idEnd = ids.end(); id != idEnd; ++id){
      const CaloCellGeometry::CornersVec& cor (cg.getSubdetectorGeometry(*id)->getGeometry(*id)->getCorners()) ;
      idToName_[id->rawId()].fillPoints(cor.begin(),cor.end());
    }
  }
}

void DumpGeom::mapRPCGeometry(const DDCompactView& cview,
			     const MuonDDDConstants& muonConstants)
{
   // filter out everythin but DT muon geometry
   std::string attribute = "ReadOutName"; 
   std::string value     = "MuonRPCHits";
   DDValue val(attribute, value, 0.0);

   // Asking only for the Muon DTs
   DDSpecificsFilter filter;
   filter.setCriteria(val,  // name & value of a variable 
		      DDSpecificsFilter::matches,
		      DDSpecificsFilter::AND, 
		      true, // compare strings otherwise doubles
		      true  // use merged-specifics or simple-specifics
		      );
   DDFilteredView fview(cview);
   fview.addFilter(filter);
   
   bool doChamber = fview.firstChild();

   // Loop on low-level "hit" detId so chamber "repeats" but map will be fine (just re-write same info more than once)
   // so this could be better in sense of faster (but do we care?) if it really was a "do chamber" kind of loop ... 
   int detid = 0;
   RPCNumberingScheme rpcnum(muonConstants);
   MuonDDDNumbering mdddnum(muonConstants);
   while (doChamber){
     // Get the Base Muon Number
     MuonBaseNumber   mbn=mdddnum.geoHistoryToBaseNumber(fview.geoHistory());
     // Get the The Rpc det Id 
     detid = 0;
     detid = rpcnum.baseNumberToUnitNumber(mbn);
     RPCDetId rpcid(detid);
     //     RPCDetId chid(rpcid.region(),rpcid.ring(),rpcid.station(),rpcid.sector(),rpcid.layer(),rpcid.subsector(),0);
     RPCDetId chid(rpcid.region(),rpcid.ring(),rpcid.station(),rpcid.sector(),rpcid.layer(),rpcid.subsector(),0);
     
     std::stringstream s;
     s << "/cms:World_1";
     DDGeoHistory::const_iterator ancestor = fview.geoHistory().begin();
     DDGeoHistory::const_iterator endancestor;
     ++ancestor; // skip the first ancestor


     /*

     NOTE: The following conditions, while (seemingly) yielding the correct
     positions and shapes for the RPC chambers, does not ultimately give the 
     correct local to global transformations for the RPC rec hits. 

     The code that follows (the line "endancestor = ...") gives the correct 
     transformation for the rec hits. 
     
     This geometry extraction will be re-worked to give correct shapes and
     transformations for all. Until then, only draw RPC rec hits.

     // in station 3 or 4 AND NOT in endcap, then fix.
     if ( ( rpcid.station() == 3 || rpcid.station() == 4 ) && std::abs(rpcid.region()) != 1 ) {
       endancestor = fview.geoHistory().end();
     } else {
       endancestor = fview.geoHistory().end() - 1;
     }
     */

     endancestor = fview.geoHistory().end();

     //      ++ancestor; // skip the first TWO ancestors
     for ( ; ancestor != endancestor; ++ ancestor )
       s << "/" << ancestor->logicalPart().name() << "_" << ancestor->copyno();
     
     std::string name = s.str();
     
     //Chamber level?      unsigned int rawid = chid.rawId();
     unsigned int rawid = rpcid.rawId();
     
     //     std::cout << idToName_.size() << " " << "RPC chamber id: " << rawid << " \tname: " << name << std::endl;
     
     //I assume that we only care to change the +1 region of the endcap (from CSCGeometryBuilderFromDDD)
     if ( rpcid.region() == 1 ) {
       DDTranslation tran    = fview.translation();
       DDRotationMatrix rota = fview.rotation();//.Inverse();
       Surface::PositionType pos(tran.x()/cm,tran.y()/cm, tran.z()/cm);
       //       std::cout << tran << std::endl;
       //       std::cout << fview.geoHistory().back().absTranslation() << std::endl;
       DD3Vector x, y, z;
       rota.GetComponents(x,y,z);
       Surface::RotationType rot (float(x.X()),float(x.Y()),float(x.Z()),
				  float(y.X()),float(y.Y()),float(y.Z()),
				  float(z.X()),float(z.Y()),float(z.Z())); 
       //       std::cout << rawid << " before: " << std::endl << rot << std::endl;
       //       std::cout << "ddd" << rota;
       //only to get ALL outputted.       if ( rpcid.region() == 1 ) {    
       //Change of axes for the forward
       Basic3DVector<float> newX(1.,0.,0.);
       Basic3DVector<float> newY(0.,0.,1.);
       if (tran.z() > 0. ) {
	 newY *= -1;
	 //	 DDRotationMatrix rotb(x.X(), y.X(), z.X(), x.Z(), y.Z(), z.Z(), -y.X(), -y.Y(), -z.Z());
	 DDRotationMatrix rotb(x.X(), z.X(), -y.X(), x.Y(), z.Y(), -y.Y(), x.Z(), z.Z(), -y.Z()); 
	 //	 std::cout <<" transformed dd: " << rotb << std::endl;
       } else {
	 //	 DDRotationMatrix rotb(x.X(), y.X(), z.X(), x.Y(), y.Y(), z.Y(), x.Z(), y.Z(), z.Z()); 
	 DDRotationMatrix rotb(x.X(), z.X(), y.X(), x.Y(), z.Y(), y.Y(), x.Z(), z.Z(), y.Z()); 
	 //	 std::cout <<" transformed dd: " << rotb << std::endl;
       }
       Basic3DVector<float> newZ(0.,1.,0.);
       rot.rotateAxes (newX, newY,newZ);

       //       std::cout << "after: " << std::endl << rot << std::endl;

//        std::cout << " new dd: " << std::endl;
//        std::cout << rot.xx() << ", " << rot.yx() << ", " << rot.zx() << std::endl;
//        std::cout << rot.xy() << ", " << rot.yy() << ", " << rot.zy() << std::endl;
//        std::cout << rot.xz() << ", " << rot.yz() << ", " << rot.zz() << std::endl;
       Basic3DVector<float> thetran(tran.X(), tran.Y(), tran.Z());
       thetran = rot * thetran;
       //       std::cout << thetran.x() << ", " << thetran.y() << ", " << thetran.z() << std::endl;
     }      
     
     idToName_[rawid] = Info(name);
     //      std::cout << " " << idToName_.size() << std::endl;
     
     doChamber = fview.nextSibling(); // go to next chamber
   }
}


// ------------ method called to for each event  ------------
void
DumpGeom::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   std::cout << "In the DumpGeom::analyze method..." << std::endl;
   using namespace edm;

   ESTransientHandle<TGeoManager> geoh;
   iSetup.get<DisplayGeomRecord>().get(geoh);
   TGeoManager *geom = const_cast<TGeoManager*>(geoh.product());

   int level = 1 + geom->GetTopVolume()->CountNodes(100, 3);

   std::cout << "In the DumpGeom::analyze method...obtained main geometry, level="
             << level << std::endl;

   
   ESTransientHandle<DDCompactView> viewH;
   iSetup.get<IdealGeometryRecord>().get(viewH);

   edm::ESHandle<MuonDDDConstants> mdc;
   iSetup.get<MuonNumberingRecord>().get(mdc);

   edm::ESHandle<GeometricDet> rDD;
   iSetup.get<IdealGeometryRecord>().get( rDD );

   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<CaloGeometryRecord>().get(pG);     

   edm::ESHandle<DTGeometry> muonDTGeom;
   iSetup.get<MuonGeometryRecord>().get(muonDTGeom);     

//    if ( pG.isValid() ) {
//      std::cout << "pG is valid" << std::endl;
//    } else {
//      std::cout << "pG is NOT valid" << std::endl;
//    }

   mapDTGeometry(*viewH, *mdc, *muonDTGeom);
   std::cout << "In the DumpGeom::analyze method...done with DT" << std::endl;
   mapCSCGeometry(*viewH, *mdc);
   std::cout << "In the DumpGeom::analyze method...done with CSC" << std::endl;
//    for ( std::map<unsigned int, Info>::const_iterator it = idToName_.begin();
// 	 it != idToName_.end(); ++ it ) {
//      CSCDetId cscdetid (it->first);
//      std::cout << "CSCDetId: " << cscdetid << " " << it->first << " " << it->second.name <<  std::endl;
//    }
   mapTrackerGeometry(*viewH, *rDD);
   std::cout << "In the DumpGeom::analyze method...done with Tracker" << std::endl;
   mapEcalGeometry(*viewH, *pG);
   std::cout << "In the DumpGeom::analyze method...done with Ecal" << std::endl;
   mapRPCGeometry(*viewH, *mdc);
   std::cout << "In the DumpGeom::analyze method...done with RPC" << std::endl;
   
   // TCanvas * canvas = new TCanvas( );
   // top->Draw("ogle");
   // std::stringstream s;
   // s<<"dump"<<level<<".eps";
   // canvas->SaveAs(s.str().c_str());
   // delete canvas;

   // Matevz, 7.7.2010
   // This is needed to avoid errors from TGeoManager::cd() that cause process termination.
   // They occur in the loop below (ecal) when extraction level is too low.
   // The list is long ... Muons, Ecal, maybe something else.
   // Run like this to see:
   //   cmsRun dump_cfg.py 2>&1 | less
   ErrorHandlerFunc_t old_eh = SetErrorHandler(DefaultErrorHandler);

   std::stringstream s2;
    s2<<"cmsGeom"<<level<<".root";
   TFile f(s2.str().c_str(),"RECREATE");
   
   TTree* tree = new TTree("idToGeo","Raw detector id association with geomtry");
   UInt_t v_id;
   TString* v_path(new TString);
   char v_name[1000];
   Float_t v_vertex[24];
   TGeoHMatrix* v_matrix(new TGeoHMatrix);
   // TGeoVolume* v_volume(new TGeoVolume);
   // TObject* v_shape(new TObject);
   
   tree->SetBranchStyle(0);
   tree->Branch("id",&v_id,"id/i");
   // tree->Branch("path","TString",&v_path);
   tree->Branch("path",&v_name,"path/C");
   // tree->Branch("matrix","TGeoHMatrix",&v_matrix);
   // tree->Branch("volume","TGeoVolume",&v_volume);
   // tree->Branch("shape","TObject",&v_shape);
   tree->Branch("points",&v_vertex,"points[24]/F");
   for ( std::map<unsigned int, Info>::const_iterator itr = idToName_.begin();
	 itr != idToName_.end(); ++itr )
     {
	v_id = itr->first;
	*v_path = itr->second.name.c_str();
	for(unsigned int i=0; i<24; ++i) v_vertex[i]=itr->second.points[i];
	strcpy(v_name,itr->second.name.c_str());
	geom->cd(*v_path);
	v_matrix = geom->GetCurrentMatrix();
	// v_volume = geom->GetCurrentVolume();
	// v_shape = geom->GetCurrentVolume()->GetShape();
	tree->Fill();
     }
   f.WriteTObject(geom);
   f.WriteTObject(tree);
   f.Close();

   // MT -- goes with the above work-around.
   SetErrorHandler(old_eh);
}

// ------------ method called once each job just before starting event loop  ------------
void 
DumpGeom::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DumpGeom::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(DumpGeom);
