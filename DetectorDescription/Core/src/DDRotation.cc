
#include <cmath>
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "CLHEP/Units/SystemOfUnits.h"

#include <sstream>
#include <cstdlib>

// Message logger.
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//static DDRotationMatrix GLOBAL_UNIT;

//DDBase<DDName,DDRotationMatrix*>::StoreT::pointer_type 
//  DDBase<DDName,DDRotationMatrix*>::StoreT::instance_ = 0;

std::ostream & operator<<(std::ostream & os, const DDRotation & r)
{
  DDBase<DDName,DDRotationMatrix*>::def_type defined(r.isDefined());
  if (defined.first) {
    os << *(defined.first) << " ";
    if (defined.second) {
      const DDRotationMatrix & rm = *(r.rotation());
      os << "t=" << rm.axis().theta()/deg << "deg "
         << "p=" << rm.axis().phi()/deg << "deg "
	 << "a=" << rm.delta()/deg << "deg";
      DCOUT_V('R', rm);
    }
    else {
      os << "* rotation not defined * ";  
    }
  }  
  else {
    os << "* rotation not declared * ";  
  }  
  return os;
}


DDRotation::DDRotation() : DDBase<DDName,DDRotationMatrix*>()
{
  //static bool onlyOnce=true;
  //if (onlyOnce) {
//  static DDRotationMatrix rm_;
  prep_ = StoreT::instance().create(DDName("",""),new DDRotationMatrix);

}


DDRotation::DDRotation(const DDName & name) : DDBase<DDName,DDRotationMatrix*>()
{
   prep_ = StoreT::instance().create(name);

}


DDRotation::DDRotation(const DDName & name, DDRotationMatrix * rot)
 : DDBase<DDName,DDRotationMatrix*>()
{
  prep_ = StoreT::instance().create(name,rot);

}


DDRotation::DDRotation(DDRotationMatrix * rot)
 : DDBase<DDName,DDRotationMatrix*>()
{
  prep_ = StoreT::instance().create(rot);

}

void DDRotation::clear()
{
  StoreT::instance().clear();
}

/*
const DDRotationMatrix * DDRotation::rotation() const { return &(rep()); }

DDRotationMatrix * DDRotation::rotation() { return &(rep()); }
*/

/*
DDRotationMatrix * DDRotaton::unit() 
{
  static DDRotationMatrix r_unit_; 
  return &r_unit_;
}
*/

DDRotation DDrot(const DDName & ddname, DDRotationMatrix * rot)
{
   // memory of rot goes sto DDRotationImpl!!
   //DCOUT('c', "DDrot: new rotation " << ddname);
   //if (rot) rot->invert();
   return DDRotation(ddname, rot);
}
 

DDRotation DDrot(const DDName & ddname,
                         double thetaX, double phiX,
			 double thetaY, double phiY,
			 double thetaZ, double phiZ)
{
  //   DDRotationMatrix * rot = 0;
   
   // define 3 unit std::vectors
   Hep3Vector x(cos(phiX)*sin(thetaX), sin(phiX)*sin(thetaX), cos(thetaX));
   Hep3Vector y(cos(phiY)*sin(thetaY), sin(phiY)*sin(thetaY), cos(thetaY));
   Hep3Vector z(cos(phiZ)*sin(thetaZ), sin(phiZ)*sin(thetaZ), cos(thetaZ));
   
   double tol = 1.0e-3; // Geant4 compatible
   double check = (x.cross(y))*z; // in case of a LEFT-handed orthogonal system 
                                  // this must be -1
   if ((1.-check)>tol) {
     edm::LogError("DDRotation") << ddname << " is not a RIGHT-handed orthonormal matrix!" << std::endl;
     throw DDException( ddname.name() + std::string(" is not RIGHT-handed!" ) );
   }
//    if ((1.-fabs(check))>tol) {
//      ostd::stringstream o;
//      o << "matrix is not an (left or right handed) orthonormal matrix! (in deg)" << std::endl
//        << " thetaX=" << thetaX/deg << " phiX=" << phiX/deg << std::endl
//        << " thetaY=" << thetaY/deg << " phiY=" << phiY/deg << std::endl
//        << " thetaZ=" << thetaZ/deg << " phiZ=" << phiZ/deg << std::endl;
//      edm::LogError("DDRotation") << o.str() << std::endl;

//      throw DDException( o.str() );
//    }
   /**WAS:   HepRep3x3 temp(x.x(),x.y(),x.z(),
                            y.x(),y.y(),y.z(),
		            z.x(),z.y(),z.z()); //matrix representation
     IS NOW:*/

   HepRep3x3 temp(x.x(),y.x(),z.x(),
                  x.y(),y.y(),z.y(),
		  x.z(),y.z(),z.z()); //matrix representation

   //   rot = new DDRotationMatrix(temp);
   return DDRotation(ddname, new DDRotationMatrix(temp));  
   
}
 
   
DDRotation DDrotReflect(const DDName & ddname, DDRotationMatrix * rot)
{
   // memory of rot goes sto DDRotationImpl!!
   //DCOUT('c', "DDrot: new rotation " << ddname);
//    if (rot) rot->invert();
   return DDRotation(ddname, rot);
}

DDRotation DDrotReflect(const DDName & ddname,
                         double thetaX, double phiX,
			 double thetaY, double phiY,
			 double thetaZ, double phiZ)
{
  //   DDRotationMatrix * rot = 0;
   
   // define 3 unit std::vectors forming the new left-handed axes 
   Hep3Vector x(cos(phiX)*sin(thetaX), sin(phiX)*sin(thetaX), cos(thetaX));
   Hep3Vector y(cos(phiY)*sin(thetaY), sin(phiY)*sin(thetaY), cos(thetaY));
   Hep3Vector z(cos(phiZ)*sin(thetaZ), sin(phiZ)*sin(thetaZ), cos(thetaZ));
   
   double tol = 1.0e-3; // Geant4 compatible
   double check = (x.cross(y))*z; // in case of a LEFT-handed orthogonal system 
                                  // this must be -1
   if ((1.+check)>tol) {
     edm::LogError("DDRotation") << ddname << " is not a LEFT-handed orthonormal matrix!" << std::endl;
     throw DDException( ddname.name() + std::string(" is not LEFT-handed!" ) );
   }
   
   // Now create a DDRotationMatrix (HepRotation), which IS left handed. 
   // This is not forseen in CLHEP, but can be achieved using the
   // constructor which does not check its input arguments!
   
   /**WAS:   HepRep3x3 temp(x.x(),x.y(),x.z(),
                            y.x(),y.y(),y.z(),
		            z.x(),z.y(),z.z()); //matrix representation
     IS NOW:*/

   HepRep3x3 temp(x.x(),y.x(),z.x(),
                  x.y(),y.y(),z.y(),
		  x.z(),y.z(),z.z()); //matrix representation

   //   rot = new DDRotationMatrix(temp);
   
   //DCOUT('c', "DDrotReflect: new reflection " << ddname);
   //rot->invert();
   return DDRotation(ddname, new DDRotationMatrix(temp));  
   				  		   		  			 
}		


DDRotationMatrix * DDcreateRotationMatrix(double thetaX, double phiX,
			 double thetaY, double phiY,
			 double thetaZ, double phiZ)
{
   // define 3 unit std::vectors forming the new left-handed axes 
   Hep3Vector x(cos(phiX)*sin(thetaX), sin(phiX)*sin(thetaX), cos(thetaX));
   Hep3Vector y(cos(phiY)*sin(thetaY), sin(phiY)*sin(thetaY), cos(thetaY));
   Hep3Vector z(cos(phiZ)*sin(thetaZ), sin(phiZ)*sin(thetaZ), cos(thetaZ));
   
   double tol = 1.0e-3; // Geant4 compatible
   double check = (x.cross(y))*z; // in case of a LEFT-handed orthogonal system 
                                  // this must be -1, RIGHT-handed: +1
   if ((1.-fabs(check))>tol) {
     std::ostringstream o;
     o << "matrix is not an (left or right handed) orthonormal matrix! (in deg)" << std::endl
       << " thetaX=" << thetaX/deg << " phiX=" << phiX/deg << std::endl
       << " thetaY=" << thetaY/deg << " phiY=" << phiY/deg << std::endl
       << " thetaZ=" << thetaZ/deg << " phiZ=" << phiZ/deg << std::endl;
     edm::LogError("DDRotation") << o.str() << std::endl;
     
     
     throw DDException( o.str() );
   }
   
   // Now create a DDRotationMatrix (HepRotation), which IS left handed. 
   // This is not forseen in CLHEP, but can be achieved using the
   // constructor which does not check its input arguments!
   
   /**WAS:   HepRep3x3 temp(x.x(),x.y(),x.z(),
                            y.x(),y.y(),y.z(),
		            z.x(),z.y(),z.z()); //matrix representation
     IS NOW:*/

   HepRep3x3 temp(x.x(),y.x(),z.x(),
                  x.y(),y.y(),z.y(),
		  x.z(),y.z(),z.z()); //matrix representation

   return new DDRotationMatrix(temp);   
}			 

							 							 
DDRotation DDanonymousRot(DDRotationMatrix * rot)
{
  return DDRotation(rot);
}
