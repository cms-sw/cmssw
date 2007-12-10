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
// $Id$
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
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "TTree.h"

//
// class decleration
//

class DumpGeom : public edm::EDAnalyzer {
   public:
      explicit DumpGeom(const edm::ParameterSet&);
      ~DumpGeom();


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
      
      std::cout << "chamber id: " << rawid << " \tname: " << name << std::endl;
      
      idToName_[rawid] = name;
      
      doChamber = fview.nextSibling(); // go to next chamber
   }
}


// ------------ method called to for each event  ------------
void
DumpGeom::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   ESHandle<DDCompactView> viewH;
   iSetup.get<IdealGeometryRecord>().get(viewH);

   edm::ESHandle<MuonDDDConstants> mdc;
   iSetup.get<MuonNumberingRecord>().get(mdc);
   
   std::auto_ptr<TGeoManager> geom(new TGeoManager("cmsGeo","CMS Detector"));
   //NOTE: the default constructor does not create the identity matrix
   if(0==gGeoIdentity) {
      gGeoIdentity = new TGeoIdentity("Identity");
   }

   //Who owns this stuff?
   TGeoMaterial* matVacuum = new TGeoMaterial("Vacuum");
   TGeoMedium* vacuum = new TGeoMedium("Vacuum",1,matVacuum);


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
	 for(int i=0; i<parentStack.size();++i) {
	    std::cout <<" ";
	 }
	 std::cout << info.first.name()<<" "<<info.second->copyno_<<" "
		   << DDSolidShapesName::name(info.first.solid().shape())<<std::endl;
      }

      bool childAlreadyExists = (0 != nameToVolume_[info.first.name()]);
      TGeoVolume* child = createVolume(std::string(info.first.name()),
				       info.first.solid(),
				       vacuum);
      if(0!=child) {
	 //add to parent
	 parentStack.back()->AddNode(child,
				 info.second->copyno_,
				 createPlacement(info.second->rotation(),
						 info.second->translation()));
	 child->SetLineColor(kBlue);
      }
      if(0 == child || childAlreadyExists || level_ == parentStack.size()) {
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
   mapDTGeometry(*viewH, *mdc);

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
