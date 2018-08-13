#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "DetectorDescription/Core/interface/Store.h"
#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDName.h"

namespace DDI {
  class Material;
}

// internal usage
bool DDCheckMaterial(DDMaterial& mip, std::pair<std::string,DDName> & result, int rlevel = 0)
{
   std::string no_composites = " NO-COMPOSITES ";
   std::string no_density    = " NO-DENSITY ";
   std::string costit_nok    = " CONSTITUENT NOK ";
   std::string no_z = " NO-Z ";
   std::string no_a = " NO-A "; 
      
      std::string curr_err = "";
      bool err = false;
      
      if (mip.isDefined().first == nullptr) {
        err=true;
	curr_err += "material not declared; unknown material!";
	//edm::LogError("DDCheckMaterials") << "material not declared!" << std::endl; //exit(1);
	result.first = curr_err;
	return err;
      }
      
      if (mip.isDefined().second == 0) {
        err=true;
	curr_err += "material name=" + mip.name().ns() + ":" + mip.name().name() 
	            + " is declared but not defined";
	result.first = curr_err;
	return err;	    
      }

      DDMaterial & mp = mip;
      result.second=mp.ddname();	 
      
      if (!mp.density()) {
        err=true;
	curr_err += no_density;
      }
      
      if ( ( !mp.z() || !mp.a() ) && !mp.noOfConstituents() ) {
        err=true;
	curr_err += no_z;
	curr_err += "or";
	curr_err += no_a;
      }	
      
      if ( ( !mp.z() && !mp.a() ) && !mp.noOfConstituents() ) {
        err=true;
	curr_err += no_composites;
      }
      
      if ( !mp.z() && !mp.a() && !mp.density() && !mp.noOfConstituents() ){
        err=true;
	curr_err = " NOT-DEFINED ";
      } 	
      
      if (err) {
        result.first=curr_err;
      }	
      
      // recursive checking of constituents
      // the composite material is not ok if only one constituent is not ok
      int loop = mp.noOfConstituents() - 1; 
      
      for (; loop>=0; --loop) { 
        std::pair<std::string,DDName> res("","");
	DDMaterial mat(mp.ddname()); // bit slow but comfortable ...
	DDMaterial mimpl = mat.constituent(loop).first;
	++rlevel; // recursion level
	bool c_err = DDCheckMaterial(mimpl,res, rlevel);
	if (c_err) {
	  err = err | c_err;
	  curr_err = curr_err + std::string(" constituents have errors:\n") + std::string(4*rlevel,' ') 
		   + std::string(" ") + res.first;
	  result.first=curr_err;	   
	}
	--rlevel;
      }
      
      return err;
}


//! Checks all registered materials and sends a report /p os
bool DDCheckMaterials(std::ostream & os, std::vector<std::pair<std::string,DDName> > * res)
{
   bool result = false;
   std::vector<std::pair<std::string,DDName> > errors;
   
   auto& mr = DDBase<DDName, std::unique_ptr<DDI::Material>>::StoreT::instance();

   for( const auto& i : mr ) {
	std::pair<std::string,DDName> error("","");
	DDMaterial tmat(i.first); 

	if (DDCheckMaterial(tmat,error)) {
	   errors.emplace_back(error);
	}	      
   }

   std::string s(" ");   
   os << "[DDCore:Report] Materials " << std::endl;
   os << s << mr.size() << " Materials declared" << std::endl;
   os << s << "detected errors:" << errors.size() << std::endl;
   for( auto j : errors ) {
     os << std::endl << s << j.second << "  " << j.first << std::endl;
     result = true;
   }
   if(res) *res = errors;
   return result;
}
