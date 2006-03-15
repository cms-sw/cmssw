#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerLevelBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerLevelBuilder_H

#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerAbstractConstruction.h"
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
    bool operator()(const GeometricDet* a, const GeometricDet* b) const{
      return a->type() <
	b->type();
    }
  };
  
  struct LessZ{
    bool operator()(const GeometricDet* a, const GeometricDet* b)
    {
      return a->translation().z() < b->translation().z();   
    }
  };
  
  struct LessModZ{
    bool operator()(const GeometricDet* a, const GeometricDet* b)
    {
      return fabs(a->translation().z()) < fabs(b->translation().z());   
    }
  };
  
  
  struct ExtractPhi:public uFcn{
    double operator()(const GeometricDet* a)const{
      const double pi = 3.141592653592;
      double phi = a->translation().phi();
      return( phi>= 0 ? phi:phi+2*pi);   
    }
  };

  struct ExtractPhiModule:public uFcn{
    double operator()(const GeometricDet* a)const{
      const double pi = 3.141592653592;
      std::vector<const GeometricDet*> comp = a->components().front()->components();
      float phi = 0.;
      for(unsigned int i=0;i<comp.size();i++){
	double phi1 = comp[i]->translation().phi() >= 0 ? comp[i]->translation().phi(): 
	  comp[i]->translation().phi()+2*pi; 
	phi+= phi1;
      }
      double com = comp.front()->translation().phi() >= 0 ? comp.front()->translation().phi():
	2*pi + comp.front()->translation().phi();
      double temp = fabs(phi/float(comp.size()) - com) > 2. ? 
	pi - phi/float(comp.size()):
	phi/float(comp.size());
      temp = temp >= 0? temp:2*pi+temp;
      return temp;
    }
  };
  
  struct LessR_module{
    bool operator()(const GeometricDet* a, const GeometricDet* b)
    {
      
      return a->components().front()->translation().perp() < 
	b->components().front()->translation().perp();
      
    }
  };
  
  struct LessR{
    bool operator()(const GeometricDet* a, const GeometricDet* b)
    {
      
      return a->translation().perp() < b->translation().perp();
      
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
