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
// $Id: DumpGeom.cc,v 1.24 2010/03/22 20:08:19 case Exp $
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

      TGeoShape* createShape(const std::string& iName,
			     const DDSolid& iSolid);
      TGeoVolume* createVolume(const std::string& iName,
			       const DDSolid& iSolid,
			       const DDMaterial& iMaterial);

   TGeoMaterial* createMaterial(const DDMaterial& iMaterial);

      void mapDTGeometry(const DDCompactView& cview,
			 const MuonDDDConstants& muonConstants);
      void mapCSCGeometry(const DDCompactView& cview,
			  const MuonDDDConstants& muonConstants);
      void mapTrackerGeometry(const DDCompactView& cview,
			      const GeometricDet& gd);
      void mapEcalGeometry(const DDCompactView& cview,
                           const CaloGeometry& cg);
      void mapRPCGeometry(const DDCompactView& cview,
                          const MuonDDDConstants& muonConstants);

      // ----------member data ---------------------------
      int level_;
      bool verbose_;
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

      std::map<std::string, TGeoShape*>    nameToShape_;
      std::map<std::string, TGeoVolume*>   nameToVolume_;
      std::map<std::string, TGeoMaterial*> nameToMaterial_;
      std::map<std::string, TGeoMedium*>   nameToMedium_;
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
  //  std::cout << "in createPlacement" << std::endl;
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
      //      std::cout <<"  shape "<<iSolid<<std::endl;
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
	    
	    std::auto_ptr<TGeoShape> left( createShape(boolSolid.solidA().name().fullname(),
						       boolSolid.solidA()) );
	    std::auto_ptr<TGeoShape> right( createShape(boolSolid.solidB().name().fullname(),
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
	    
	    std::auto_ptr<TGeoShape> left( createShape(boolSolid.solidA().name().fullname(),
						       boolSolid.solidA()) );
	    std::auto_ptr<TGeoShape> right( createShape(boolSolid.solidB().name().fullname(),
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
	    
	    std::auto_ptr<TGeoShape> left( createShape(boolSolid.solidA().name().fullname(),
						       boolSolid.solidA()) );
	    std::auto_ptr<TGeoShape> right( createShape(boolSolid.solidB().name().fullname(),
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
		       const DDMaterial& iMaterial)
{
   TGeoVolume* v=nameToVolume_[iName];
   if( 0==v) {
   
      TGeoShape* solid = createShape(iSolid.name().fullname(),
				     iSolid);
      std::string mat_name = iMaterial.name().fullname();
      TGeoMedium *geo_med  = nameToMedium_[mat_name];
      if (geo_med == 0)
      {
         TGeoMaterial *geo_mat = createMaterial(iMaterial);
         geo_med = new TGeoMedium(mat_name.c_str(), 0, geo_mat);
         nameToMedium_[mat_name] = geo_med;
      }
      if (solid)
      {
	 v = new TGeoVolume(iName.c_str(),
			    solid,
			    geo_med);
      }
      nameToVolume_[iName]=v;
   }
   return v;
}

TGeoMaterial*
DumpGeom::createMaterial(const DDMaterial& iMaterial)
{
   std::string   mat_name = iMaterial.name().fullname();
   TGeoMaterial *mat      = nameToMaterial_[mat_name];

   if (mat == 0)
   {
      if (iMaterial.noOfConstituents() > 0)
      {
         TGeoMixture *mix = new TGeoMixture(mat_name.c_str(),
                                            iMaterial.noOfConstituents(),
                                            iMaterial.density()*cm3/g);
         for (int i = 0; i < iMaterial.noOfConstituents(); ++i)
         {
            mix->AddElement(createMaterial(iMaterial.constituent(i).first),
                            iMaterial.constituent(i).second);
         }
         mat = mix;
      }
      else
      {
         mat = new TGeoMaterial(mat_name.c_str(),
                                iMaterial.a()*mole/g, iMaterial.z(),
                                iMaterial.density()*cm3/g);
      }
      nameToMaterial_[mat_name] = mat;
   }

   return mat;
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
      
      idToName_[rawid] = Info(name);
      
      doChamber = fview.nextSibling(); // go to next chamber
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
     // in station 3 or 4 AND NOT in endcap, then fix.
     if ( ( rpcid.station() == 3 || rpcid.station() == 4 ) && std::abs(rpcid.region()) != 1 ) {
       endancestor = fview.geoHistory().end();
     } else {
       endancestor = fview.geoHistory().end() - 1;
     }
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

   ESHandle<DDCompactView> viewH;
   iSetup.get<IdealGeometryRecord>().get(viewH);

   edm::ESHandle<MuonDDDConstants> mdc;
   iSetup.get<MuonNumberingRecord>().get(mdc);

   edm::ESHandle<GeometricDet> rDD;
   iSetup.get<IdealGeometryRecord>().get( rDD );

   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<CaloGeometryRecord>().get(pG);     

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

   std::cout << "about to initialize the DDCompactView walker" << std::endl;
   DDCompactView::walker_type walker(viewH->graph());
   DDCompactView::walker_type::value_type info = 
      walker.current();
   //The top most item is actually the volume holding both the
   // geometry AND the magnetic field volumes!
   walker.firstChild();

   TGeoVolume* top = createVolume(info.first.name().fullname(),
				  info.first.solid(),
                                  info.first.material());

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

   // Matevz, 23.3.2010
   // This is needed to avoid errors from TGeo to cause process termination.
   // The root patch will be submitted for integration in 3.6.0-pre4.
   ErrorHandlerFunc_t old_eh = SetErrorHandler(DefaultErrorHandler);

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

      bool childAlreadyExists = (0 != nameToVolume_[info.first.name().fullname()]);
      TGeoVolume* child = createVolume(info.first.name().fullname(),
				       info.first.solid(),
				       info.first.material());
      if(0!=child && info.second != 0) {
	 parentStack.back()->AddNode(child,
				 info.second->copyno_,
				 createPlacement(info.second->rotation(),
						 info.second->translation()));
	 child->SetLineColor(kBlue);
      }  else {
	if ( info.second == 0 ) {
	  break;
 	}
      }
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

   // MT -- goes with the above work-around.
   SetErrorHandler(old_eh);

   geom->CloseGeometry();
   std::cout << "In the DumpGeom::analyze method...done with main geometry" << std::endl;
   mapDTGeometry(*viewH, *mdc);
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
   f.WriteTObject(&*geom);
   f.WriteTObject(tree);
   f.Close();
   // geom->Export(s2.str().c_str());
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
