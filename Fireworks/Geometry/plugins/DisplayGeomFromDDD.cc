// -*- C++ -*-
//
// Package:    DisplayGeomFromDDD
// Class:      DisplayGeomFromDDD
// 
/**\class DisplayGeomFromDDD DisplayGeomFromDDD.h Fireworks/DisplayGeomFromDDD/src/DisplayGeomFromDDD.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  "Christopher Jones"
//         Created:  Thu Mar 18 16:19:17 CDT 2010
// $Id$
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"
#include "TGeoManager.h"

#include <memory>
#include <iostream>
#include <sstream>

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

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
#include "TROOT.h"
#include "TSystem.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "Math/GenVector/RotationX.h"



// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Fireworks/Geometry/interface/DisplayGeomRecord.h"



//
// class declaration
//

class DisplayGeomFromDDD : public edm::ESProducer {
   public:
      DisplayGeomFromDDD(const edm::ParameterSet&);
      ~DisplayGeomFromDDD();

      typedef boost::shared_ptr<TGeoManager> ReturnType;

      ReturnType produce(const DisplayGeomRecord&);
   private:
      // ----------member data ---------------------------
         TGeoShape* createShape(const std::string& iName,
   			     const DDSolid& iSolid);
         TGeoVolume* createVolume(const std::string& iName,
   			       const DDSolid& iSolid,
   			       TGeoMedium* iMed);
         // ----------member data ---------------------------
         int level_;
         bool verbose_;

         std::map<std::string, TGeoShape*> nameToShape_;
         std::map<std::string, TGeoVolume*> nameToVolume_;

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
DisplayGeomFromDDD::DisplayGeomFromDDD(const edm::ParameterSet& iConfig):
level_(iConfig.getUntrackedParameter<int>("level",6)),
verbose_(iConfig.getUntrackedParameter<bool>("verbose",false))
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


DisplayGeomFromDDD::~DisplayGeomFromDDD()
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
DisplayGeomFromDDD::createShape(const std::string& iName,
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
DisplayGeomFromDDD::createVolume(const std::string& iName,
		       const DDSolid& iSolid,
		       TGeoMedium* iMed) {
   TGeoVolume* v=nameToVolume_[iName];
   if( 0==v) {
   
      TGeoShape* solid = createShape(iSolid.name().fullname(),
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


// ------------ method called to produce the data  ------------
DisplayGeomFromDDD::ReturnType
DisplayGeomFromDDD::produce(const DisplayGeomRecord& iRecord)
{
   using namespace edm::es;

      edm::ESHandle<DDCompactView> viewH;
      iRecord.getRecord<IdealGeometryRecord>().get(viewH);

      boost::shared_ptr<TGeoManager> geom(new TGeoManager("cmsGeo","CMS Detector"));
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

      TGeoVolume* top = createVolume(info.first.name().fullname(),
   				  info.first.solid(),vacuum);

      if(0==top) {
         return boost::shared_ptr<TGeoManager>();
      }
      geom->SetTopVolume(top);
      //ROOT chokes unless colors are assigned
      top->SetVisibility(kFALSE);
      top->SetLineColor(kBlue);

      std::vector<TGeoVolume*> parentStack;
      parentStack.push_back(top);

      if( not walker.firstChild() ) {
         return boost::shared_ptr<TGeoManager>();
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

         bool childAlreadyExists = (0 != nameToVolume_[info.first.name().fullname()]);
         TGeoVolume* child = createVolume(info.first.name().fullname(),
   				       info.first.solid(),
   				       vacuum);
         //mikes debug output
   //       std::cout << "done with " << info.first.name() << " about to mess w the stack" 
   // 		<< " child = " << child 
   // 		<< " childAlreadyExist = " << childAlreadyExists
   // 		<< " level_ = " << level_ << " parentStack.size() = " << parentStack.size();
   //       std::cout << " info.second " << info.second << std::endl;
   // end mikes debug output

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
         }  else {
   	if ( info.second == 0 ) {
   //mikes debug output 	  std::cout << "OKAY! it IS 0" << std::endl;
   	  break;
    	}
   //mikes debug output
   // 	if ( parentStack.size() != 0 ) {
   //  	  std::cout << "huh?  have we popped back further than we should? and why?" << std::endl;
   //  	}
   //end mikes debug output
         }
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


      geom->CloseGeometry();

   return geom ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(DisplayGeomFromDDD);
