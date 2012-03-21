#include "DetectorDescription/Core/src/Polycone.h" 

#include <assert.h>
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include <cmath>

using DDI::Polycone;

Polycone::Polycone (double startPhi, double deltaPhi,
                    const std::vector<double> & z,
                    const std::vector<double> & rmin,
                    const std::vector<double> & rmax) : Solid (ddpolycone_rrz)	      
{
   p_.push_back(startPhi);
   p_.push_back(deltaPhi);
   if((z.size()!=rmin.size()) || (z.size()!=rmax.size()) )
   {
      throw cms::Exception("DDException") << "Polycone(..): std::vectors z,rmin,rmax not of same length";
   } 
   else
   {
      for(unsigned int i=0;i<z.size(); ++i)
      {
         p_.push_back(z[i]);
         p_.push_back(rmin[i]);
         p_.push_back(rmax[i]);
      }
   }
}	      


Polycone::Polycone (double startPhi, double deltaPhi,
                    const std::vector<double> & z,
                    const std::vector<double> & r) : Solid (ddpolycone_rz)	      
{
   p_.push_back(startPhi);
   p_.push_back(deltaPhi);
   if(z.size()!=r.size())
   {
      throw cms::Exception("DDException") << "Polycone(..): std::vectors z,rmin,rmax not of same length";
   } 
   else
   {
      for( unsigned int i = 0; i < z.size(); ++i )
      {
         p_.push_back(z[i]);
         p_.push_back(r[i]);
      }
   }
}	     

double Polycone::volume() const 
{
   double result = -1.;
   if (shape_==ddpolycone_rrz) 
   {
      unsigned int loop = (p_.size()-2)/3 -1;
      assert(loop>0);
      double sec=0;
      DCOUT('V',"Polycone::volume(), loop=" << loop);
      int i=2;
      for (unsigned int j=2; j<(loop+2); ++j) {
         double dz= std::fabs(p_[i]-p_[i+3]);
         DCOUT('v', "   dz=" << dz/cm << "cm zi=" << p_[i] << " zii=" << p_[i+3] );
         double v_min = dz * pi/3. *(  p_[i+1]*p_[i+1] + p_[i+4]*p_[i+4]
                                     + p_[i+1]*p_[i+4] );
         DCOUT('v', "   v_min" << v_min/cm3 << "cm3 rmi=" << p_[i+1]);		    
         double v_max = dz * pi/3. *(  p_[i+2]*p_[i+2] + p_[i+5]*p_[i+5]
                                     + p_[i+2]*p_[i+5] );
         DCOUT('v', "   v_max" << v_max/cm3 << "cm3");		    
         double s = v_max - v_min;
         //assert(s>=0);
         sec += s;
         i += 3;			    			    					   
      }  
      result = sec * std::fabs(p_[1])/rad/(2.*pi);
   }
   
   if (shape_==ddpolycone_rz) 
   {
      double volume=0;
      double phiFrom=p_[0]/rad;
      double phiTo=(p_[0]+p_[1])/rad;
      double slice=(std::fabs(phiFrom-phiTo))/(2*pi);
      double zBegin=0;
      double zEnd=0;
      double rBegin=0;
      double rEnd=0;
      double z=0;
      unsigned int i=2;
      
      while(i<(p_.size()-2))
      {
         zBegin=p_[i];
         zEnd=p_[i+2];
         rBegin=p_[i+1];
         rEnd=p_[i+3];
         z=zBegin-zEnd;
         
         /* for calculation of volume1 look at calculation of DDConsImpl of a volume1. Furthermore z can be smaller than zero. This makes sense since volumes we have to substract */
         double volume1=(rEnd*rEnd+rBegin*rBegin+rBegin*rEnd)*z/3;
         volume=volume+volume1;
         i=i+2;
      }
      
      /* last line (goes from last z/r value to first */
      i=p_.size()-2;
      zBegin=p_[i];
      zEnd=p_[2];
      rBegin=p_[i+1];
      rEnd=p_[3];
      z=zBegin-zEnd;
      
      double volume2=(rEnd*rEnd+rBegin*rBegin+rBegin*rEnd)*z/3;
      volume=volume+volume2;
      volume=std::fabs(slice*pi*volume);
      result = volume;
   }
   return result;
}
