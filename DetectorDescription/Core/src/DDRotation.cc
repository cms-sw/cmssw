#include <cmath>
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <Math/AxisAngle.h>

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
      DDAxisAngle   ra(rm);
      os << "t=" << ra.Axis().Theta()/deg << "deg "
         << "p=" << ra.Axis().Phi()/deg << "deg "
	 << "a=" << ra.Angle()/deg << "deg"; 
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
  //  static DDRotationMatrix* rm_ = new DDRotationMatrix;
  //  prep_ = StoreT::instance().create(DDName("",""), rm_ );
  constexpr char const* baseName = "DdBlNa";
  static int countBlank;
  char buf[64];
  snprintf(buf, 64, "%s%i", baseName, countBlank++);
  prep_ = StoreT::instance().create(DDName(buf,baseName), new DDRotationMatrix );
  //  std::cout << "making a BLANK " << buf << " named rotation, " << prep_->second << std::endl;
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
  static std::string baseNoName("DdNoNa");
  static int countNN;
  static std::ostringstream ostr2;
  ostr2 << countNN++;
  prep_ = StoreT::instance().create(DDName(baseNoName+ostr2.str(), baseNoName), rot);
  //  std::cout << "making a NO-NAME " << baseNoName+ostr2.str() << " named rotation, " << prep_->second << std::endl;
  ostr2.clear();
  ostr2.str("");
}

// void DDRotation::clear()
// {
//   StoreT::instance().clear();
// }

DDRotation DDrot(const DDName & ddname, DDRotationMatrix * rot)
{
   // memory of rot goes sto DDRotationImpl!!
   //DCOUT('c', "DDrot: new rotation " << ddname);
   //if (rot) rot->invert();
   return DDRotation(ddname, rot);
}
 
// makes sure that the DDRotationMatrix constructed is right-handed and orthogonal.
DDRotation DDrot(const DDName & ddname,
                         double thetaX, double phiX,
			 double thetaY, double phiY,
			 double thetaZ, double phiZ)
{
   // define 3 unit std::vectors
   DD3Vector x(cos(phiX)*sin(thetaX), sin(phiX)*sin(thetaX), cos(thetaX));
   DD3Vector y(cos(phiY)*sin(thetaY), sin(phiY)*sin(thetaY), cos(thetaY));
   DD3Vector z(cos(phiZ)*sin(thetaZ), sin(phiZ)*sin(thetaZ), cos(thetaZ));
   
   double tol = 1.0e-3; // Geant4 compatible
   double check = (x.Cross(y)).Dot(z); // in case of a LEFT-handed orthogonal system this must be -1
   if (fabs(1.-check)>tol) {
     edm::LogError("DDRotation") << ddname << " is not a RIGHT-handed orthonormal matrix!" << std::endl;
     throw cms::Exception("DDException") << ddname.name() << " is not RIGHT-handed!";
   }

   DDRotationMatrix* rot = new DDRotationMatrix(x.x(),y.x(),z.x(),
						x.y(),y.y(),z.y(),
						x.z(),y.z(),z.z());

   return DDRotation(ddname, rot);  
   
}
 
   
DDRotation DDrotReflect(const DDName & ddname, DDRotationMatrix * rot)
{
   // memory of rot goes sto DDRotationImpl!!
   //DCOUT('c', "DDrot: new rotation " << ddname);
//    if (rot) rot->invert();
   return DDRotation(ddname, rot);
}


// makes sure that the DDRotationMatrix built is LEFT-handed coordinate system (i.e. reflected)
DDRotation DDrotReflect(const DDName & ddname,
                         double thetaX, double phiX,
			 double thetaY, double phiY,
			 double thetaZ, double phiZ)
{
   
   // define 3 unit std::vectors forming the new left-handed axes 
   DD3Vector x(cos(phiX)*sin(thetaX), sin(phiX)*sin(thetaX), cos(thetaX));
   DD3Vector y(cos(phiY)*sin(thetaY), sin(phiY)*sin(thetaY), cos(thetaY));
   DD3Vector z(cos(phiZ)*sin(thetaZ), sin(phiZ)*sin(thetaZ), cos(thetaZ));
   
   double tol = 1.0e-3; // Geant4 compatible
   double check = (x.Cross(y)).Dot(z); // in case of a LEFT-handed orthogonal system this must be -1
   if (fabs(1.+check)>tol) {
     edm::LogError("DDRotation") << ddname << " is not a LEFT-handed orthonormal matrix!" << std::endl;
     throw cms::Exception("DDException") << ddname.name() << " is not LEFT-handed!";
   }
   
   DDRotationMatrix* rot = new DDRotationMatrix(x.x(),y.x(),z.x(),
						x.y(),y.y(),z.y(),
						x.z(),y.z(),z.z());

   //DCOUT('c', "DDrotReflect: new reflection " << ddname);
   //rot->invert();
   return DDRotation(ddname, rot);  
   				  		   		  			 
}		


// does NOT check LEFT or Right handed coordinate system takes either.
DDRotationMatrix * DDcreateRotationMatrix(double thetaX, double phiX,
			 double thetaY, double phiY,
			 double thetaZ, double phiZ)
{
   // define 3 unit std::vectors forming the new left-handed axes 
   DD3Vector x(cos(phiX)*sin(thetaX), sin(phiX)*sin(thetaX), cos(thetaX));
   DD3Vector y(cos(phiY)*sin(thetaY), sin(phiY)*sin(thetaY), cos(thetaY));
   DD3Vector z(cos(phiZ)*sin(thetaZ), sin(phiZ)*sin(thetaZ), cos(thetaZ));
   
   double tol = 1.0e-3; // Geant4 compatible
   double check = (x.Cross(y)).Dot(z);// in case of a LEFT-handed orthogonal system this must be -1, RIGHT-handed: +1
   if ((1.-fabs(check))>tol) {
     std::ostringstream o;
     o << "matrix is not an (left or right handed) orthonormal matrix! (in deg)" << std::endl
       << " thetaX=" << thetaX/deg << " phiX=" << phiX/deg << std::endl
       << " thetaY=" << thetaY/deg << " phiY=" << phiY/deg << std::endl
       << " thetaZ=" << thetaZ/deg << " phiZ=" << phiZ/deg << std::endl;
     edm::LogError("DDRotation") << o.str() << std::endl;
     
     
     throw cms::Exception("DDException") << o.str();
   }
   
   return new DDRotationMatrix(x.x(),y.x(),z.x(),
			       x.y(),y.y(),z.y(),
			       x.z(),y.z(),z.z());
}			 

							 							 
DDRotation DDanonymousRot(DDRotationMatrix * rot)
{
  return DDRotation(rot);
}
