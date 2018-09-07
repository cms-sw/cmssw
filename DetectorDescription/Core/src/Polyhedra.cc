#include "DetectorDescription/Core/src/Polyhedra.h" 
#include "DetectorDescription/Core/interface/DDUnits.h"

#include <cmath>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/src/Solid.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using DDI::Polyhedra;

using std::fabs;
using std::cos;
using std::sin;
using namespace dd;
using namespace dd::operators;

Polyhedra::Polyhedra( int sides, double startPhi, double deltaPhi,
                      const std::vector<double> & z,
                      const std::vector<double> & rmin,
                      const std::vector<double> & rmax)
  : Solid(DDSolidShape::ddpolyhedra_rrz)	      
{
  p_.emplace_back(sides);
  p_.emplace_back(startPhi);
  p_.emplace_back(deltaPhi);
  if((z.size()!=rmin.size()) || (z.size()!=rmax.size()) )
  {
    throw cms::Exception("DDException") << "Polyhedra(..): std::vectors z,rmin,rmax not of same length";
  } 
  else
  {
    for(unsigned int i=0;i<z.size(); ++i)
    {
      p_.emplace_back(z[i]);
      p_.emplace_back(rmin[i]);
      p_.emplace_back(rmax[i]);
    }
  }
}	      

Polyhedra::Polyhedra( int sides, double startPhi, double deltaPhi,
                      const std::vector<double> & z,
                      const std::vector<double> & r) : Solid(DDSolidShape::ddpolyhedra_rz)	      
{
  p_.emplace_back(sides);
  p_.emplace_back(startPhi);
  p_.emplace_back(deltaPhi);
  if(z.size()!=r.size())
  {
    throw cms::Exception("DDException") << "Polyhedra(..): std::vectors z,rmin,rmax not of same length";
  } 
  else
  {
    for(unsigned int i=0;i<z.size(); ++i)
    {
      p_.emplace_back(z[i]);
      p_.emplace_back(r[i]);
    }
  }
}	     

double Polyhedra::volume() const
{
  double volume = 0;
  /* the following assumption is made:
     there are at least 3 eaqual sides
     if there is a complete circle (this has to be done,
     otherwise you can not define a polygon in a circle */
  
  /* the calculation for the volume is similar as in
     the case of the polycone. However, the rotation
     is not defined as part of a circle, but as sides
     in a regular polygon (specified by parameter "sides").
     The sides are defined betwee startPhi and endPhi and
     form triangles within the circle they are defined in.
     First we need to determine the aread of side.
     let alpha |startPhi-endPhi|.
     the half the angle of 1 side is beta=0.5*(alph/sides).
     If r is the raddius of the circle in which the regular polygon is defined,
     the are of such a side will be
     0.5*(height side)*(base side)=0.5*(cos(beta)*r)*(2*sin(beta)*r)= cos(beta)sin(beta)*r*r.
     r is the radius that varies if we "walk" over the boundaries of
     the polygon that is described by the z and r values
     (this yields the same integral primitive as used with the Polycone.
     Read Polycone documentation in code first if you do not understand this */
   
  //FIXME: rz, rrz !!
  if (shape()==DDSolidShape::ddpolyhedra_rrz) 
  {
    int loop = (p_.size()-3)/3 -1;
    double sec=0;
    double a = 0.5*fabs(CONVERT_TO( p_[2], rad ) / p_[0]);
    int i=3;
    for (int j=3; j<(loop+3); ++j) 
    {
      double dz= fabs(p_[i]-p_[i+3]);
      /*
	double ai,  aii;
	ai  = (p_[i+2]*p_[i+2] - p_[i+1]*p_[i+1]);
	aii = (p_[i+5]*p_[i+5] - p_[i+4]*p_[i+4]);
	//double s = dz/3.*(ai*bi + 0.5*(ai*bii + bi*aii) + aii*bii);
	double s = dz/3.*sin(a)*cos(a)*(ai + aii + 0.5*(ai+aii));
      */
      double z=dz/2.;
      
      double H1=(p_[i+2]-p_[i+1])*cos(a);
      double Bl1=p_[i+1]*sin(a);
      double Tl1=p_[i+2]*sin(a);
      
      double H2=(p_[i+5]-p_[i+4])*cos(a);
      double Bl2=p_[i+4]*sin(a);
      double Tl2=p_[i+5]*sin(a);
      
      double s = (2*H1*Bl1+2*H1*Tl1)*z+(H1*Bl2-2*H1*Bl1+H1*Tl2-2*H1*Tl1+H2*Bl1+H2*Tl1+H2*Tl2-H2*Tl1)*z+(2/3)*(H2*Bl2-H2*Bl1-H1*Bl2+H1*Bl1-H1*Tl2+H1*Tl1)*z; 
      s = s*p_[0];
      sec += s;
      i+=3;
    }
    volume=sec;
    return volume;
  }  
  int sides=int(p_[0]);
  //double phiFrom=p_[1]/rad;
  double phiDelta=CONVERT_TO( p_[2], rad );
  
  double zBegin=0;
  double zEnd=0;
  double rBegin=0;
  double rEnd=0;
  double z=0;
  double alpha=0;
  double beta=0;
  unsigned int i=3;
   
  alpha=fabs(phiDelta);
  beta=0.5*(alpha/sides);
  
  while(i<(p_.size()-2))
  {
    zBegin=p_[i];
    zEnd=p_[i+2];
    rBegin=p_[i+1];
    rEnd=p_[i+3];
    z=zBegin-zEnd;
      
    /* volume for 1 side (we multiplie by cos(beta)sin(beta)sides later*/
    double volume1=(rEnd*rEnd+rBegin*rBegin+rBegin*rEnd)*z/3;
    
    volume=volume+volume1;
      
    i=i+2;
  }
   
  /* last line (goes from last z/r value to first */
   
  i=p_.size()-2;
  zBegin=p_[i];
  zEnd=p_[3];
  rBegin=p_[i+1];
  rEnd=p_[4];
  z=zBegin-zEnd;
   
  double volume2=(rEnd*rEnd+rBegin*rBegin+rBegin*rEnd)*z/3;
  
  volume=volume+volume2;
  
  volume=fabs(sides*cos(beta)*sin(beta)*volume);
  
  return volume;
}

void DDI::Polyhedra::stream(std::ostream & os) const
{
  os << " sides=" << p_[0]
     << " startPhi[deg]=" << CONVERT_TO( p_[1], deg )
     << " dPhi[deg]=" << CONVERT_TO( p_[2], deg )
     << " Sizes[cm]=";
  for (unsigned k=3; k<p_.size(); ++k)
    os << CONVERT_TO( p_[k], cm ) << " ";
}
