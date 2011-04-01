//
#include <vector>
#include "DetectorDescription/Core/interface/DDMaterial.h"

// Message logger.

// internal usage
bool DDCheckMaterial(DDMaterial& mip, std::pair<std::string,DDName> & result)
{
   std::string no_composites = " NO-COMPOSITES ";
   std::string no_density    = " NO-DENSITY ";
   std::string costit_nok    = " CONSTITUENT NOK ";
   std::string no_z = " NO-Z ";
   std::string no_a = " NO-A "; 
   static int rlevel = 0;
      
      std::string curr_err = "";
      bool err = false;
      
      if (mip.isDefined().first == 0) {
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
/*
     else {
        edm::LogInfo << " material name=" << flush 
	     << *mip.isDefined().first << std::endl;
      }
*/      
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
	bool c_err = DDCheckMaterial(mimpl,res);
	if (c_err) {
	  err = err | c_err;
	  curr_err = curr_err + std::string(" constituents have errors:\n") + std::string(4*rlevel,' ') 
	           //+ res.second.ns() + std::string(":") + res.second.name() 
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
   
   
   //DDMaterialReg::instance_t& mr = DDMaterialReg::instance();
   //DDMaterialReg::instance_t::iterator i = mr.begin();
   typedef DDBase<DDName,DDI::Material*>::StoreT RegT;
   RegT::value_type& mr = RegT::instance();
   RegT::value_type::iterator i = mr.begin();
   //edm::LogError("DDCheckMaterials") << " material checking, registry access, exiting! " << std::endl; exit(1);
   for(; i != mr.end(); ++i) {
	std::pair<std::string,DDName> error("","");
	DDMaterial tmat(i->first); 
	//exit(1);
	if (DDCheckMaterial(tmat,error)) {
	   errors.push_back(error);
	}	      
   }

   std::string s(" ");   
   os << "[DDCore:Report] Materials " << std::endl;
   os << s << mr.size() << " Materials declared" << std::endl;
   os << s << "detected errors:" << errors.size() << std::endl;
   std::vector<std::pair<std::string,DDName> >::iterator j = errors.begin();
   for (;j!=errors.end();++j) {
     os << std::endl << s << j->second << "  " << j->first << std::endl;
     result = true;
   }
   if(res) *res = errors;
   return result;
}
