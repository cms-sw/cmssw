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
// $Id: DumpGeom.cc,v 1.7 2008/02/13 00:07:02 case Exp $
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

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

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

#include "CLHEP/Units/SystemOfUnits.h"
#include "Math/GenVector/RotationX.h"

///////////////////////////////////////////////////////////
// Muons

#include <Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/DTGeometry/interface/DTChamber.h>
#include <Geometry/DTGeometry/interface/DTLayer.h>

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "Geometry/MuonNumbering/interface/DTNumberingScheme.h"
#include <Geometry/CSCGeometry/src/CSCWireGroupPackage.h>
#include "Geometry/MuonNumbering/interface/CSCNumberingScheme.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include "Geometry/Records/interface/MuonNumberingRecord.h"

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

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
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include "TTree.h"

//
// class decleration
//

class DumpGeom : public edm::EDAnalyzer {

   public:
      explicit DumpGeom(const edm::ParameterSet&);
      ~DumpGeom();

  template <class T> friend class CaloGeometryLoader;//<EcalBarralGeometry>;

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      TGeoShape* createShape(const std::string& iName,
			     const DDSolid& iSolid);
      TGeoVolume* createVolume(const std::string& iName,
			       const DDSolid& iSolid,
			       TGeoMedium* iMed);
      void mapDTGeometry(const DDCompactView& cview,
			 const MuonDDDConstants& muonConstants);
      void mapCSCGeometry(const DDCompactView& cview,
			 const MuonDDDConstants& muonConstants);
      void mapTrackerGeometry(const DDCompactView& cview,
			      const GeometricDet& gd);
      void mapEcalGeometry(const DDCompactView& cview,
			      const CaloGeometry& cg);

      // ----------member data ---------------------------
      int level_;
      bool verbose_;

      std::map<std::string, TGeoShape*> nameToShape_;
      std::map<std::string, TGeoVolume*> nameToVolume_;
      std::map<unsigned int, std::string> idToName_;
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
DumpGeom::DumpGeom(const edm::ParameterSet& iConfig):
   level_(iConfig.getUntrackedParameter<int>("level",4)),
   verbose_(iConfig.getUntrackedParameter<bool>("verbose",false))
{
   //now do what ever initialization is needed

}


DumpGeom::~DumpGeom()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//
static
TGeoCombiTrans* createPlacement(const DDRotationMatrix& iRot,
				const DDTranslation& iTrans)
{
  std::cout << "in createPlacement" << std::endl;
   double elements[9];
   iRot.GetComponents(elements);
   TGeoRotation r;
   r.SetMatrix(elements);

   TGeoTranslation t(iTrans.x()/cm,
		     iTrans.y()/cm,
		     iTrans.z()/cm);

   return new TGeoCombiTrans(t,r);
}


TGeoShape* 
DumpGeom::createShape(const std::string& iName,
	    const DDSolid& iSolid)
{
   TGeoShape* rSolid= nameToShape_[iName];
   if(0==rSolid) {
   
      const std::vector<double>& params = iSolid.parameters();
      std::cout <<"  shape "<<iSolid<<std::endl;
      switch(iSolid.shape()) {
	 case ddbox:
	    rSolid = new TGeoBBox(
	       iName.c_str(),
	       params[0]/cm,
	       params[1]/cm,
	       params[2]/cm);
	    break;
	 case ddcons:
	    rSolid = new TGeoConeSeg(
	       iName.c_str(),
	       params[0]/cm,
	       params[1]/cm,
	       params[2]/cm,
	       params[3]/cm,
	       params[4]/cm,
	       params[5]/deg,
	       params[6]/deg+params[5]/deg
	       );
	    break;
	 case ddtubs:
	    //Order in params is  zhalf,rIn,rOut,startPhi,deltaPhi
	    rSolid= new TGeoTubeSeg(
	       iName.c_str(),
	       params[1]/cm,
	       params[2]/cm,
	       params[0]/cm,
	       params[3]/deg,
	       params[4]/deg);
	    break;
	 case ddtrap:
	    rSolid =new TGeoTrap(
	       iName.c_str(),
	       params[0]/cm,  //dz
	       params[1]/deg, //theta
	       params[2]/deg, //phi
	       params[3]/cm,  //dy1
	       params[4]/cm,  //dx1
	       params[5]/cm,  //dx2
	       params[6]/deg, //alpha1
	       params[7]/cm,  //dy2
	       params[8]/cm,  //dx3
	       params[9]/cm,  //dx4
	       params[10]/deg);//alpha2
	    break;
	 case ddpolycone_rrz:	 
	    rSolid = new TGeoPcon(
	       iName.c_str(),
	       params[0]/deg,
	       params[1]/deg,
	       (params.size()-2)/3) ;
	    {
	       std::vector<double> temp(params.size()+1);
	       temp.reserve(params.size()+1);
	       temp[0]=params[0]/deg;
	       temp[1]=params[1]/deg;
	       temp[2]=(params.size()-2)/3;
	       std::copy(params.begin()+2,params.end(),temp.begin()+3);
	       for(std::vector<double>::iterator it=temp.begin()+3;
		   it != temp.end();
		   ++it) {
		  *it /=cm;
	       }	       
	       rSolid->SetDimensions(&(*(temp.begin())));
	    }
	    break;
	 case ddpolyhedra_rrz:
	    rSolid = new TGeoPgon(
	       iName.c_str(),
	       params[1]/deg,
	       params[2]/deg,
	       static_cast<int>(params[0]),
	       (params.size()-3)/3);
	    {
	       std::vector<double> temp(params.size()+1);
	       temp[0]=params[1]/deg;
	       temp[1]=params[2]/deg;
	       temp[2]=params[0];
	       temp[3]=(params.size()-3)/3;
	       std::copy(params.begin()+3,params.end(),temp.begin()+4);
	       for(std::vector<double>::iterator it=temp.begin()+4;
		   it != temp.end();
		   ++it) {
		  *it /=cm;
	       }
	       rSolid->SetDimensions(&(*(temp.begin())));
	    }
	    break;
	 case ddpseudotrap:
	 {
	    //implementation taken from SimG4Core/Geometry/src/DDG4SolidConverter.cc
	    static DDRotationMatrix s_rot(ROOT::Math::RotationX(90.*deg));
	    DDPseudoTrap pt(iSolid);
	    assert(pt.radius() < 0);
	    double x=0;
	    double r = fabs(pt.radius());
	    if( pt.atMinusZ()) {
	       x=pt.x1();
	    } else {
	       x=pt.x2();
	    }
	    double openingAngle = 2.0*asin(x/r);
	    double h=pt.y1()<pt.y2()? pt.y2() :pt.y1();
	    h+=h/20.;
	    double displacement=0;
	    double startPhi = 0;
	    double delta = sqrt((r+x)*(r-x));
	    if(pt.atMinusZ()) {
	       displacement=-pt.halfZ() - delta;
	       startPhi = 270.-openingAngle/deg/2.0;
	    }else {
	       displacement = pt.halfZ() + delta;
	       startPhi = 90. - openingAngle/deg/2.;
	    }
	    std::auto_ptr<TGeoShape> trap( new TGeoTrd2(pt.name().name().c_str(),
							pt.x1()/cm,
							pt.x2()/cm,
							pt.y1()/cm,
							pt.y2()/cm,
							pt.halfZ()/cm) );
	    std::auto_ptr<TGeoShape> tubs( new TGeoTubeSeg(pt.name().name().c_str(),
							   0.,
							   r/cm,
							   h/cm,
							   startPhi,
							   openingAngle) );
	    TGeoSubtraction* sub = new TGeoSubtraction(trap.release(),
						       tubs.release(),
						       createPlacement(s_rot,
								       DDTranslation(0.,
										     0.,
										     displacement)));
	    rSolid = new TGeoCompositeShape(iName.c_str(),
					    sub);
	    
	    
	    break;
	 }
	 case ddsubtraction:
	 {
	    DDBooleanSolid boolSolid(iSolid);
	    if(!boolSolid) {
	       throw cms::Exception("GeomConvert") <<"conversion to DDBooleanSolid failed";
	    }
	    
	    std::auto_ptr<TGeoShape> left( createShape(boolSolid.solidA().name(),
						       boolSolid.solidA()) );
	    std::auto_ptr<TGeoShape> right( createShape(boolSolid.solidB().name(),
							boolSolid.solidB()));
	    if( 0 != left.get() &&
		0 != right.get() ) {
	       TGeoSubtraction* sub = new TGeoSubtraction(left.release(),right.release(),
							  gGeoIdentity,
							  createPlacement(
							     *(boolSolid.rotation().matrix()),
							     boolSolid.translation()));
	       rSolid = new TGeoCompositeShape(iName.c_str(),
					       sub);
	    }
	    break;
	 }
	 case ddunion:
	 {
	    DDBooleanSolid boolSolid(iSolid);
	    if(!boolSolid) {
	       throw cms::Exception("GeomConvert") <<"conversion to DDBooleanSolid failed";
	    }
	    
	    std::auto_ptr<TGeoShape> left( createShape(boolSolid.solidA().name(),
						       boolSolid.solidA()) );
	    std::auto_ptr<TGeoShape> right( createShape(boolSolid.solidB().name(),
							boolSolid.solidB()));
	    //DEBUGGING
	    //break;
	    if( 0 != left.get() &&
		0 != right.get() ) {
	       TGeoUnion* boolS = new TGeoUnion(left.release(),right.release(),
						gGeoIdentity,
						createPlacement(
						   *(boolSolid.rotation().matrix()),
						   boolSolid.translation()));
	       rSolid = new TGeoCompositeShape(iName.c_str(),
					       boolS);
	    }
	    break;
	 }
	 case ddintersection:
	 {
	    DDBooleanSolid boolSolid(iSolid);
	    if(!boolSolid) {
	       throw cms::Exception("GeomConvert") <<"conversion to DDBooleanSolid failed";
	    }
	    
	    std::auto_ptr<TGeoShape> left( createShape(boolSolid.solidA().name(),
						       boolSolid.solidA()) );
	    std::auto_ptr<TGeoShape> right( createShape(boolSolid.solidB().name(),
							boolSolid.solidB()));
	    if( 0 != left.get() &&
		0 != right.get() ) {
	       TGeoIntersection* boolS = new TGeoIntersection(left.release(),
							      right.release(),
							      gGeoIdentity,
							      createPlacement(
								 *(boolSolid.rotation().matrix()),
								 boolSolid.translation()));
	       rSolid = new TGeoCompositeShape(iName.c_str(),
					       boolS);
	    }
	    break;
	 }
	 default:
	    break;
      }
      nameToShape_[iName]=rSolid;
   }
   if(0==rSolid) {
      std::cerr <<"COULD NOT MAKE "<<iName<<std::endl;
   }
   return rSolid;
}


TGeoVolume* 
DumpGeom::createVolume(const std::string& iName,
		       const DDSolid& iSolid,
		       TGeoMedium* iMed) {
   TGeoVolume* v=nameToVolume_[iName];
   if( 0==v) {
   
      TGeoShape* solid = createShape(iSolid.name(),
				     iSolid);
      if (solid) {
	 v = new TGeoVolume(iName.c_str(),
			    solid,
			    iMed);
      }
      nameToVolume_[iName]=v;
   }
   return v;
}

void DumpGeom::mapDTGeometry(const DDCompactView& cview,
			     const MuonDDDConstants& muonConstants)
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
      
      idToName_[rawid] = name;
      
      doChamber = fview.nextSibling(); // go to next chamber
   }
}

/** 
 ** By Michael Case
 ** method mapCSCGeometry(...)
 ** date: 01-25-2008
 ** Description:
 **   This is a hack to do the following.  Assign layer det id's to 
 ** a DD "path" or "geo History".  Because the current loop in the analyze
 ** does not iterate below the Polycarb panel of the geometry, the detId
 ** for the layer is mapped to the chamber's path in the DD.  This means
 ** there is no way for the user of the produced root file to determine
 ** which hit is in which layer.  Any hit position will not be "translatable"
 ** to a position within the chamber, i.e. no local track segments can be 
 ** displayed if the user wants that level of detail.
 ** 
 **/
void DumpGeom::mapCSCGeometry(const DDCompactView& cview,
			     const MuonDDDConstants& muonConstants) {
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

  //  std::cout << "****************** doSubDets is";
  //  if (doSubDets) std::cout << " TRUE"; else std::cout << " FALSE";
  //  std::cout << "*******************" << std::endl;
  // Loop on chambers
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
    // The above gives you the CHAMBER DetID but not the LAYER.
    // For CSCs the layers are built from specpars of the chamber.
    // I will try to do the same here without actually building
    // chamber geometry (i.e. won't copy whole of CSCGeometryBuilderFromDDD

     idToName_[chamberId.rawId()] = name;
     std::cout << "CSC chamber: " << chamberId.rawId() << " \tname: " << name << std::endl;
     
     //  If it's ME11 you need to have two detId's per chamber. This is how to construct the detId
     //  copied from the CSCGeometryBuilder code.
     int jstation = chamberId.station();
     int jring    = chamberId.ring();
     int jchamber = chamberId.chamber();
     int jendcap  = chamberId.endcap();
     if ( jstation==1 && jring==1 ) {
       CSCDetId detid1a = CSCDetId( jendcap, 1, 4, jchamber, 0 );
       std::cout << "CSC chamber: " << detid1a.rawId() << " \tname: " << name << std::endl;
       idToName_[detid1a.rawId()] = name;
     }
/* We don't need layers for now, till we get geometry for them fixed
    int jend   = chamberId.endcap();
    int jstat  = chamberId.station();
    int jring  = chamberId.ring();
    int jch    = chamberId.chamber();

    // Create the component layers of this chamber   
    // We're taking the z as the z of the wire plane within the layer (middle of gas gap)

    // Specify global z of layer by offsetting from centre of chamber: since layer 1 
    // is nearest to IP in stations 1/2 but layer 6 is nearest in stations 3/4, 
    // we need to adjust sign of offset appropriately...
    int localZwrtGlobalZ = +1;
    if ( (jend==1 && jstat<3 ) || ( jend==2 && jstat>2 ) ) localZwrtGlobalZ = -1;
    int globalZ = +1;
    if ( jend == 2 ) globalZ = -1;
    for ( short j = 1; j <= 6; ++j ) {
      CSCDetId layerId = CSCDetId( jend, jstat, jring, jch, j );

      // centre of chamber is at global z = gtran[2]
      // centre of layer j=1 is 2.5 layerSeparations from average AGV, hence centre of layer w.r.t. AF
      // NOT USED RIGHT NOW float zlayer = gtran[2] - globalZ*zAverageAGVtoAF + localZwrtGlobalZ*(3.5-j)*layerSeparation;

      unsigned int rawid = layerId.rawId();

      // COULD MODIFY name so that we have the "fake/hack" layer name since we don't know what it is at 
      // this point.  THERE MAY BE A FIX to this by iterating separately over a different filter 
      // which looks at the layers only.  Anyway, there is a real pain here in that we can not do this
      // right now anyway because of the depth (level_) limit in root software the "Woops!!!" error.
      // mf:ME11AlumFrame is the name of the chamber, then ME11 or ME1A is the layer... can not dist...
      //   names are ?  ME1A_ActiveGasVol?
      //   names are ?  ME11_ActiveGasVol?
      //  OR ME11_Layer?  I choose the layer name.
      // 	 std::cout << "fview.logicalPart().name();= " << fview.logicalPart().name() << "fview.logicalPart().name().name();" << fview.logicalPart().name().name() << std::endl;
      // 	 std::string prefName = (fview.logicalPart().name().name()).substr(0,4);
      // 	 name = baseName + "/" + prefName + "_ActiveGasVol_";
      // 	 std::ostringstream ostr;
      // 	 ostr << j;
      // 	 name += ostr.str();
      idToName_[rawid] = name;
      std::cout << "chamber id: " << rawid << " \tname: " << name << std::endl;

      // same rotation as chamber
      // same x and y as chamber
      //	 layerPosition( gtran[0], gtran[1], zlayer );


    } // layer construction within chamber
    */ 
     
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
    expv.goTo( (*git)->navType() );

    std::stringstream s;
    s << "/cms:World_1";
    DDGeoHistory::const_iterator ancestor = expv.geoHistory().begin();
    ++ancestor; // skip the first ancestor
    for ( ; ancestor != expv.geoHistory().end(); ++ ancestor )
      s << "/" << ancestor->logicalPart().name() << "_" << ancestor->copyno();
      
    std::string name = s.str();
    id = int((*git)->geographicalID());
    //    std::cout << "Tracker id: " << id << " \tname: " << name << std::endl;
    idToName_[id] = name;
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
     idToName_[tid] = name;
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
     idToName_[tid] = name;
     doSubDets = fview.nextSibling(); // go to next
    }

// //   int n=0;
//   std::vector<DetId> ids=geom->getValidDetIds(DetId::Ecal,EcalEndcap);
//   for (std::vector<DetId>::iterator i=ids.begin(); i!=ids.end(); i++) {
// //     n++;
// //     const CaloCellGeometry* cell=geom->getGeometry(*i);
// //     EEDetId closestCell= EEDetId(geom->getClosestCell(dynamic_cast<const TruncatedPyramid*>(cell)->getPosition(0.)));
// //     assert (closestCell == EEDetId(*i) );
//     unsigned int tid(*i);
//     std::vector<int> tint = geom->getDDNavType(tid);
//     DDExpandedView epv(cview);
//     epv.goTo(tint);
// //     if (tid == 872420050 || tid == 872420051 ) {
// //       std::cout << "DumpGeom::detId = " << tid ;
// //       std::cout << " fvgeohist: " << epv.geoHistory() << std::endl;
// //     }

// //    std::cout << "id: " << tid << " path: " << epv.geoHistory() << std::endl;	    
//     // build map here
//     std::stringstream s;
//     s << "/cms:World_1";
//     DDGeoHistory::const_iterator ancestor = epv.geoHistory().begin();
//     ++ancestor; // skip the first ancestor
//     for ( ; ancestor != epv.geoHistory().end(); ++ ancestor )
//       s << "/" << ancestor->logicalPart().name() << "_" << ancestor->copyno();
      
//     std::string name = s.str();
//     idToName_[tid] = name;
//   }
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
     idToName_[tid] = name;
     doSubDets = fview.nextSibling(); // go to next
    }

// //   int n=0;
//   std::vector<DetId> ids=geom->getValidDetIds(DetId::Ecal,EcalPreshower);
//   for (std::vector<DetId>::iterator i=ids.begin(); i!=ids.end(); i++) {
// //     n++;
// //     const CaloCellGeometry* cell=geom->getGeometry(*i);
// //     EEDetId closestCell= EEDetId(geom->getClosestCell(dynamic_cast<const TruncatedPyramid*>(cell)->getPosition(0.)));
// //     assert (closestCell == EEDetId(*i) );
//     unsigned int tid(*i);
//     std::vector<int> tint = geom->getDDNavType(tid);
//     DDExpandedView epv(cview);
//     epv.goTo(tint);
//     //    std::cout << "id: " << tid << " path: " << epv.geoHistory() << std::endl;	    
//     // build map here
//     std::stringstream s;
//     s << "/cms:World_1";
//     DDGeoHistory::const_iterator ancestor = epv.geoHistory().begin();
//     ++ancestor; // skip the first ancestor
//     for ( ; ancestor != epv.geoHistory().end(); ++ ancestor )
//       s << "/" << ancestor->logicalPart().name() << "_" << ancestor->copyno();
      
//     std::string name = s.str();
//     idToName_[tid] = name;
//   }
  }

}


// ------------ method called to for each event  ------------
void
DumpGeom::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::cout << "In the DumpGeom::analyze method..." << std::endl;
   using namespace edm;

   ESHandle<DDCompactView> viewH;
   iSetup.get<IdealGeometryRecord>().get(viewH);

   edm::ESHandle<MuonDDDConstants> mdc;
   iSetup.get<MuonNumberingRecord>().get(mdc);

   edm::ESHandle<GeometricDet> rDD;
   iSetup.get<IdealGeometryRecord>().get( rDD );

   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<IdealGeometryRecord>().get(pG);     

//    if ( pG.isValid() ) {
//      std::cout << "pG is valid" << std::endl;
//    } else {
//      std::cout << "pG is NOT valid" << std::endl;
//    }
   std::auto_ptr<TGeoManager> geom(new TGeoManager("cmsGeo","CMS Detector"));
   //NOTE: the default constructor does not create the identity matrix
   if(0==gGeoIdentity) {
      gGeoIdentity = new TGeoIdentity("Identity");
   }

   //Who owns this stuff?
   TGeoMaterial* matVacuum = new TGeoMaterial("Vacuum");
   TGeoMedium* vacuum = new TGeoMedium("Vacuum",1,matVacuum);

   std::cout << "about to initialize the DDCompactView walker" << std::endl;
   DDCompactView::walker_type walker(viewH->graph());
   DDCompactView::walker_type::value_type info = 
      walker.current();
   //The top most item is actually the volume holding both the
   // geometry AND the magnetic field volumes!
   walker.firstChild();

   TGeoVolume* top = createVolume(std::string(info.first.name()),
				  info.first.solid(),vacuum);

   if(0==top) {
      return;
   }
   geom->SetTopVolume(top);
   //ROOT chokes unless colors are assigned
   top->SetVisibility(kFALSE);
   top->SetLineColor(kBlue);

   std::vector<TGeoVolume*> parentStack;
   parentStack.push_back(top);

   if( not walker.firstChild() ) {
      return;
   }
   
   do {
      DDCompactView::walker_type::value_type info = 
	 walker.current();
      if(verbose_) {
	 for(unsigned int i=0; i<parentStack.size();++i) {
	    std::cout <<" ";
	 }
	 std::cout << info.first.name()<<" "<<info.second->copyno_<<" "
		   << DDSolidShapesName::name(info.first.solid().shape())<<std::endl;
      }

      bool childAlreadyExists = (0 != nameToVolume_[info.first.name()]);
      TGeoVolume* child = createVolume(std::string(info.first.name()),
				       info.first.solid(),
				       vacuum);
      std::cout << "done with " << info.first.name() << " about to mess w the stack" 
		<< " child = " << child 
		<< " childAlreadyExist = " << childAlreadyExists
		<< " level_ = " << level_ << " parentStack.size() = " << parentStack.size();
      std::cout << " info.second " << info.second << std::endl;
      if(0!=child && info.second != 0) {
	 //add to parent
	//mikes debug output
// 	std::cout << " info.second->copyno_ = " << info.second->copyno_
// 		  << std::endl;
// 	std::cout << "adding a node to the parent" << std::endl;
	//end mikes debug output
	 parentStack.back()->AddNode(child,
				 info.second->copyno_,
				 createPlacement(info.second->rotation(),
						 info.second->translation()));
	 child->SetLineColor(kBlue);
      } 
	//mikes debug output
// else {
// 	if ( info.second == 0 ) {
// 	  std::cout << "OKAY! it IS 0" << std::endl;
// 	 break;
// 	}
// 	if ( parentStack.size() != 0 ) {
// 	  std::cout << "huh?  have we popped back further than we should? and why?" << std::endl;
// 	}
//       }
	//end mikes debug output
      if(0 == child || childAlreadyExists || level_ == int(parentStack.size()) ) {
	 if(0!=child) {
	    child->SetLineColor(kRed);
	 }
	 //stop descending
	 if( not walker.nextSibling()) {
	    while(walker.parent()) {
	       parentStack.pop_back();
	       if(walker.nextSibling()) {
		  break;
	       }
	    }
	 }
      } else {
	 if( walker.firstChild() ) {
	    parentStack.push_back(child);
	 }else {	    
	    if( not walker.nextSibling() ) {
	       while(walker.parent()) {
		  parentStack.pop_back();
		  if(walker.nextSibling()) {
		     break;
		  }
	       }
	    }
	 }
      }
   } while(not parentStack.empty());

/*
   if(0==top) {
      return;
   }

   do {
      if(verbose_) {

	 for(int i=0; i<ev.geoHistory().size();++i) {
	    std::cout <<" ";
	 }
	 std::cout << ev.logicalPart().name()<<" "<<ev.copyno()<<" "
		   << DDSolidShapesName::name(ev.logicalPart().solid().shape())<<std::endl;
      }
      if(level_==ev.geoHistory().size()) {
	 TGeoVolume* v = createVolume(std::string(ev.logicalPart().name()),
				      ev.logicalPart().solid(),
				      geom.get(),
				      vacuum);
	 if(0 != v) {

	    top->AddNode(v,ev.copyno(),createPlacement(ev.rotation(),ev.translation()));
	    if(ev.logicalPart().solid().shape() == ddpolyhedra_rrz) {
	       v->SetLineColor(kGreen);
	    }
	    if(ev.logicalPart().solid().shape() == ddbox) {
	       v->SetLineColor(kBlue);
	    }
	 }
      }

   } while( ev.next() );
*/
   geom->CloseGeometry();
   std::cout << "In the DumpGeom::analyze method...done with main geometry" << std::endl;
   mapDTGeometry(*viewH, *mdc);
   std::cout << "In the DumpGeom::analyze method...done with DT" << std::endl;
   mapCSCGeometry(*viewH, *mdc);
   std::cout << "In the DumpGeom::analyze method...done with CSC" << std::endl;
   mapTrackerGeometry(*viewH, *rDD);
   std::cout << "In the DumpGeom::analyze method...done with Tracker" << std::endl;
   mapEcalGeometry(*viewH, *pG);
   std::cout << "In the DumpGeom::analyze method...done with Ecal" << std::endl;
   
   TCanvas * canvas = new TCanvas( );
   top->Draw("ogle");

   std::stringstream s;
   s<<"dump"<<level_<<".eps";
   canvas->SaveAs(s.str().c_str());
   delete canvas;

   std::stringstream s2;
   s2<<"cmsGeom"<<level_<<".root";
   TFile f(s2.str().c_str(),"RECREATE");
   
   TTree* tree = new TTree("idToGeo","Raw detector id association with geomtry");
   UInt_t v_id;
   TString* v_path(new TString);
   char v_name[1000];
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
   for ( std::map<unsigned int, std::string>::const_iterator itr = idToName_.begin();
	 itr != idToName_.end(); ++itr )
     {
	v_id = itr->first;
	*v_path = itr->second.c_str();
	strcpy(v_name,itr->second.c_str());
	geom->cd(*v_path);
	v_matrix = geom->GetCurrentMatrix();
	// v_volume = geom->GetCurrentVolume();
	// v_shape = geom->GetCurrentVolume()->GetShape();
	tree->Fill();
     }
   f.WriteTObject(&*geom);
   f.WriteTObject(tree);
   f.Close();
   // geom->Export(s2.str().c_str());
}


// ------------ method called once each job just before starting event loop  ------------
void 
DumpGeom::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DumpGeom::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(DumpGeom);
