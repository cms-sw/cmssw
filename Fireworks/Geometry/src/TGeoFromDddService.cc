// -*- C++ -*-
//
// Package:     Geometry
// Class  :     TGeoFromDddService
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Fri Jul  2 16:11:42 CEST 2010
//

// system include files

// user include files

#include "Fireworks/Geometry/interface/TGeoFromDddService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoCompositeShape.h"
#include "TGeoPcon.h"
#include "TGeoPgon.h"
#include "TGeoCone.h"
#include "TGeoBoolNode.h"
#include "TGeoTube.h"
#include "TGeoArb8.h"
#include "TGeoTrd2.h"

#include "Math/GenVector/RotationX.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TGeoFromDddService::TGeoFromDddService(const edm::ParameterSet& pset, edm::ActivityRegistry& ar) :
   m_level      (pset.getUntrackedParameter<int> ("level", 10)),
   m_verbose    (pset.getUntrackedParameter<bool>("verbose",false)),
   m_eventSetup (0),
   m_geoManager (0)
{
   ar.watchPostBeginRun(this, &TGeoFromDddService::postBeginRun);
   ar.watchPostEndRun  (this, &TGeoFromDddService::postEndRun);
}

TGeoFromDddService::~TGeoFromDddService()
{
   if (m_geoManager)
   {
      delete m_geoManager;
   }
}


//==============================================================================
// public member functions
//==============================================================================

void TGeoFromDddService::postBeginRun(const edm::Run&, const edm::EventSetup& es)
{
   printf("TGeoFromDddService::postBeginRun\n");

   m_eventSetup = &es;
}

void TGeoFromDddService::postEndRun(const edm::Run&, const edm::EventSetup&)
{
   printf("TGeoFromDddService::postEndRun\n");

   // Construction of geometry fails miserably on second attempt ...
   /*
   if (m_geoManager)
   {
      delete m_geoManager;
      m_geoManager = 0;
   }
   */
   m_eventSetup = 0;
}

TGeoManager* TGeoFromDddService::getGeoManager()
{
   if (m_geoManager == 0)
   {
      if (m_eventSetup == 0)
         edm::LogError("TGeoFromDddService") << "getGeoManager() -- EventSetup not present.\n";
      else
      {
         m_geoManager = createManager(m_level);
         if (m_geoManager == 0)
            edm::LogError("TGeoFromDddService") << "getGeoManager() -- creation failed.\n";
      }
   }
   gGeoManager = m_geoManager;
   return m_geoManager;
}


//==============================================================================
// Local helpers
//==============================================================================

namespace
{
   TGeoCombiTrans* createPlacement(const DDRotationMatrix& iRot,
                                   const DDTranslation&    iTrans)
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
}


//==============================================================================
// private member functions
//==============================================================================


TGeoManager*
TGeoFromDddService::createManager(int level)
{
   using namespace edm;

   ESTransientHandle<DDCompactView> viewH;
   m_eventSetup->get<IdealGeometryRecord>().get(viewH);

   if ( ! viewH.isValid() )
   {
      return 0;
   }

   TGeoManager *geo_mgr = new TGeoManager("cmsGeo","CMS Detector");
   // NOTE: the default constructor does not create the identity matrix
   if (gGeoIdentity == 0)
   {
      gGeoIdentity = new TGeoIdentity("Identity");
   }

   std::cout << "about to initialize the DDCompactView walker" << std::endl;
   DDCompactView::walker_type             walker(viewH->graph());
   DDCompactView::walker_type::value_type info = walker.current();

   // The top most item is actually the volume holding both the
   // geometry AND the magnetic field volumes!
   walker.firstChild();

   TGeoVolume *top = createVolume(info.first.name().fullname(),
				  info.first.solid(),
                                  info.first.material());
   if (top == 0)
   {
      
      return 0;
   }

   geo_mgr->SetTopVolume(top);
   // ROOT chokes unless colors are assigned
   top->SetVisibility(kFALSE);
   top->SetLineColor(kBlue);

   std::vector<TGeoVolume*> parentStack;
   parentStack.push_back(top);

   if( ! walker.firstChild() ) {
      return 0;
   }

   do
   {
      DDCompactView::walker_type::value_type info = walker.current();

      if (m_verbose)
      {
	 for(unsigned int i=0; i<parentStack.size();++i) {
	    std::cout <<" ";
	 }
	 std::cout << info.first.name()<<" "<<info.second->copyno_<<" "
		   << DDSolidShapesName::name(info.first.solid().shape())<<std::endl;
      }

      bool childAlreadyExists = (0 != nameToVolume_[info.first.name().fullname()]);
      TGeoVolume *child = createVolume(info.first.name().fullname(),
				       info.first.solid(),
				       info.first.material());
      if (0!=child && info.second != 0)
      {
	 parentStack.back()->AddNode(child,
				 info.second->copyno_,
				 createPlacement(info.second->rotation(),
						 info.second->translation()));
	 child->SetLineColor(kBlue);
      }
      else
      {
	if ( info.second == 0 ) {
	  break;
 	}
      }
      if (0 == child || childAlreadyExists || level == int(parentStack.size()))
      {
	 if (0!=child)
         {
	    child->SetLineColor(kRed);
	 }
	 //stop descending
	 if ( ! walker.nextSibling())
         {
	    while (walker.parent())
            {
	       parentStack.pop_back();
	       if (walker.nextSibling()) {
		  break;
	       }
	    }
	 }
      }
      else
      {
	 if (walker.firstChild())
         {
	    parentStack.push_back(child);
	 }
         else
         {	    
	    if ( ! walker.nextSibling())
            {
	       while (walker.parent())
               {
		  parentStack.pop_back();
		  if (walker.nextSibling()) {
		     break;
		  }
	       }
	    }
	 }
      }
   } while ( ! parentStack.empty());

   geo_mgr->CloseGeometry();

   geo_mgr->DefaultColors();

   nameToShape_.clear();
   nameToVolume_.clear();
   nameToMaterial_.clear();
   nameToMedium_.clear();

   return geo_mgr;
}

TGeoShape* 
TGeoFromDddService::createShape(const std::string& iName,
		      const DDSolid&     iSolid)
{
   TGeoShape* rSolid= nameToShape_[iName];
   if (rSolid == 0)
   {
      const std::vector<double>& params = iSolid.parameters();
      //      std::cout <<"  shape "<<iSolid<<std::endl;
      switch(iSolid.shape())
      {
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
                                    params[3]/deg + params[4]/deg);
	    break;
	 case ddcuttubs:
	    //Order in params is  zhalf,rIn,rOut,startPhi,deltaPhi
	    rSolid= new TGeoCtub(
				 iName.c_str(),
				 params[1]/cm,
				 params[2]/cm,
				 params[0]/cm,
				 params[3]/deg,
				 params[3]/deg + params[4]/deg,
				 params[5],params[6],params[7],
				 params[8],params[9],params[10]);
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
	    const static DDRotationMatrix s_rot(ROOT::Math::RotationX(90.*deg));
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
   if (rSolid == 0)
   {
      std::cerr <<"COULD NOT MAKE "<<iName<<std::endl;
   }
   return rSolid;
}

TGeoVolume* 
TGeoFromDddService::createVolume(const std::string& iName,
		       const DDSolid& iSolid,
		       const DDMaterial& iMaterial)
{
   TGeoVolume* v=nameToVolume_[iName];
   if (v == 0)
   {
      TGeoShape* solid     = createShape(iSolid.name().fullname(),
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
      nameToVolume_[iName] = v;
   }
   return v;
}

TGeoMaterial*
TGeoFromDddService::createMaterial(const DDMaterial& iMaterial)
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

//
// const member functions
//

//
// static member functions
//
