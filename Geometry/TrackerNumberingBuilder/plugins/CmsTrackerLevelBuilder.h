#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerLevelBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerLevelBuilder_H 

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerAbstractConstruction.h"
#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerStringToEnum.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>
/**
 * Abstract Class to construct a Level in the hierarchy
 */

typedef std::unary_function<const GeometricDet*, double> uFcn;

class CmsTrackerLevelBuilder : public CmsTrackerAbstractConstruction {
public:
  virtual void build(DDFilteredView& , GeometricDet*, std::string);
  virtual ~CmsTrackerLevelBuilder(){}
  
  
  struct subDetByType{
    bool operator()(const GeometricDet* a, const GeometricDet* b) const {
      return a->type() < b->type();
    }
  };
  
  // NP** Phase2 BarrelEndcap
  struct PhiSortNP{
    bool operator()(const GeometricDet* a,const GeometricDet* b) const {
      if ( fabs(a->translation().rho() - b->translation().rho()) < 0.01 &&
           (fabs(a->translation().phi() - b->translation().phi()) < 0.01 ||
            fabs(a->translation().phi() - b->translation().phi()) > 6.27 ) &&
           a->translation().z() * b->translation().z() > 0.0 ) {
        return ( fabs(a->translation().z()) < fabs(b->translation().z()) );
      }
      else
        return false;
    }
  };


  struct LessZ{
    bool operator()(const GeometricDet* a, const GeometricDet* b) const
    {
      // NP** change for Phase 2 Tracker
      if (a->translation().z() == b->translation().z())
        {return a->translation().rho() < b->translation().rho();}
      else{
      // Original version
      return a->translation().z() < b->translation().z();}   
    }
  };
  
  struct LessModZ{
    bool operator()(const GeometricDet* a, const GeometricDet* b) const
    {
      return fabs(a->translation().z()) < fabs(b->translation().z());   
    }
  };
  
  struct LessModExtPhase2Z{
    bool operator()(const GeometricDet* a, const GeometricDet* b) const
    {
      std::string det_name_a = a->name();
      std::string det_name_b = b->name();
      if ( (det_name_a.find("PixelForwardDisk") < det_name_a.size()) &&
	   (det_name_b.find("PixelForwardDisk") < det_name_b.size()) )
	{
	  // both are inner pixels
	  // sort by z
	  return fabs(a->translation().z()) < fabs(b->translation().z());  
	}
      else if ( !(det_name_a.find("PixelForwardDisk") < det_name_a.size()) &&
		!(det_name_b.find("PixelForwardDisk") < det_name_b.size()) )
	{
	  // both are outer tracker
	  // sort by z
	  return fabs(a->translation().z()) < fabs(b->translation().z());
	}
      else
	{
	  if ( det_name_a.find("PixelForwardDisk") < det_name_a.size() )
	    {
	      // a is inner pixel
	      // let it be first
	      return true;
	    }
	  else
	    {
	      // b is inner pixel
	      // let it be first
	      return false;
	    }
	}
    }
  };

  struct ExtractPhi:public uFcn{
    double operator()(const GeometricDet* a)const{
      const double pi = 3.141592653592;
      double phi = a->phi();
      return( phi>= 0 ? phi:phi+2*pi);   
    }
  };
  
  struct ExtractPhiModule:public uFcn{
    double operator()(const GeometricDet* a)const{
      const double pi = 3.141592653592;
      std::vector<const GeometricDet*> const & comp = a->components().back()->components();
      float phi = 0.;
      bool sum = true;
      
      for(unsigned int i=0;i<comp.size();i++){
	if(fabs(comp[i]->phi())>pi/2.) { 
	  sum = false;
	  break;
	}
      }
      
      if(sum){
	for(unsigned int i=0;i<comp.size();i++){
	  phi+= comp[i]->phi();
	}
	
	double temp = phi/float(comp.size()) < 0. ? 
	  2*pi + phi/float(comp.size()):
	  phi/float(comp.size());
	return temp;
	
      }else{
	for(unsigned int i=0;i<comp.size();i++){
	  double phi1 = comp[i]->phi() >= 0 ? comp[i]->phi(): 
	    comp[i]->phi()+2*pi; 
	  phi+= phi1;
	}
	
	double com = comp.front()->phi() >= 0 ? comp.front()->phi():
	  2*pi + comp.front()->phi();
	double temp = fabs(phi/float(comp.size()) - com) > 2. ? 
	  pi - phi/float(comp.size()):
	  phi/float(comp.size());
	temp = temp >= 0? temp:2*pi+temp;
	return temp;
      }
    }
  };
  
  struct ExtractPhiGluedModule:public uFcn{
    double operator()(const GeometricDet* a)const{
      const double pi = 3.141592653592;
      std::vector<const GeometricDet*> comp;
      a->deepComponents(comp);
      float phi = 0.;
      bool sum = true;
      
      for(unsigned int i=0;i<comp.size();i++){
	if(fabs(comp[i]->phi())>pi/2.) {
	  sum = false;
	  break;
	}
      }
      
      if(sum){
	for(unsigned int i=0;i<comp.size();i++){
	  phi+= comp[i]->phi();
	}
	
	double temp = phi/float(comp.size()) < 0. ? 
	  2*pi + phi/float(comp.size()):
	  phi/float(comp.size());
	return temp;
	
      }else{
	for(unsigned int i=0;i<comp.size();i++){
	  double phi1 = comp[i]->phi() >= 0 ? comp[i]->phi(): 
	    comp[i]->translation().phi()+2*pi; 
	  phi+= phi1;
	}
	
	double com = comp.front()->phi() >= 0 ? comp.front()->phi():
	  2*pi + comp.front()->phi();
	double temp = fabs(phi/float(comp.size()) - com) > 2. ? 
	  pi - phi/float(comp.size()):
	  phi/float(comp.size());
	temp = temp >= 0? temp:2*pi+temp;
	return temp;
      }
    }
  };
  
  struct ExtractPhiMirror:public uFcn{
    double operator()(const GeometricDet* a)const{
      const double pi = 3.141592653592;
      double phi = a->phi();
      phi = (phi>= 0 ? phi : phi+2*pi); // (-pi,pi] --> [0,2pi)
      return ( (pi-phi) >= 0 ? (pi-phi) : (pi-phi)+2*pi ); // (-pi,pi] --> [0,2pi)
    }
  };
  
  struct ExtractPhiModuleMirror:public uFcn{
    double operator()(const GeometricDet* a)const{
      const double pi = 3.141592653592;
      double phi = ExtractPhiModule()(a); // [0,2pi)
      phi = ( phi <= pi ? phi : phi-2*pi );   // (-pi,pi]   
      return (pi-phi);
    }
  };
  
  struct ExtractPhiGluedModuleMirror:public uFcn{
    double operator()(const GeometricDet* a)const{
      const double pi = 3.141592653592;
      double phi = ExtractPhiGluedModule()(a); // [0,2pi)
      phi = ( phi <= pi ? phi : phi-2*pi );   // (-pi,pi]   
      return (pi-phi);
    }
  };
  
  struct LessR_module{
    bool operator()(const GeometricDet* a, const GeometricDet* b) const
    {
      return a->deepComponents().front()->rho() < 
	b->deepComponents().front()->rho();      
    }
  };
  
  struct LessR{
    bool operator()(const GeometricDet* a, const GeometricDet* b) const
    {
      return a->rho() < b->rho(); 
    }
  };
  
  
 private:
  virtual void buildComponent(DDFilteredView& , GeometricDet*, std::string) = 0;
protected:
  CmsTrackerStringToEnum theCmsTrackerStringToEnum;
private:
  virtual void sortNS(DDFilteredView& , GeometricDet*){}
  CmsTrackerStringToEnum _CmsTrackerStringToEnum;
};

#endif
