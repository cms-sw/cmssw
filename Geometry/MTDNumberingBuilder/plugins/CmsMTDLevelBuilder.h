#ifndef Geometry_MTDNumberingBuilder_CmsMTDLevelBuilder_H
#define Geometry_MTDNumberingBuilder_CmsMTDLevelBuilder_H 

#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDAbstractConstruction.h"
#include "Geometry/MTDNumberingBuilder/interface/CmsMTDStringToEnum.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Abstract Class to construct a Level in the hierarchy
 */

class CmsMTDLevelBuilder : public CmsMTDAbstractConstruction {
public:
  void build(DDFilteredView& , GeometricTimingDet*, std::string) override;
  ~CmsMTDLevelBuilder() override{}
  
  
  static bool subDetByType(const GeometricTimingDet* a, const GeometricTimingDet* b) {
    return a->type() < b->type(); // it relies on the fact that the GeometricTimingDet::GDEnumType enumerators used to identify the subdetectors in the upgrade geometries are equal to the ones of the present detector + n*100
  }
  
  // NP** Phase2 BarrelEndcap
  bool PhiSortNP(const GeometricTimingDet* a,const GeometricTimingDet* b) const {
    if ( fabs(a->translation().rho() - b->translation().rho()) < 0.01 &&
         (fabs(a->translation().phi() - b->translation().phi()) < 0.01 ||
          fabs(a->translation().phi() - b->translation().phi()) > 6.27 ) &&
         a->translation().z() * b->translation().z() > 0.0 ) {
      return ( fabs(a->translation().z()) < fabs(b->translation().z()) );
    }
    else
      return false;
  }


  static bool LessZ(const GeometricTimingDet* a, const GeometricTimingDet* b)
  {
    // NP** change for Phase 2 Tracker
    if (a->translation().z() == b->translation().z())
      {return a->translation().rho() < b->translation().rho();}
    else{
    // Original version
    return a->translation().z() < b->translation().z();}   
  }
  
  static bool LessModZ(const GeometricTimingDet* a, const GeometricTimingDet* b)
  {
    return fabs(a->translation().z()) < fabs(b->translation().z());   
  }
  

  static double extractPhi(const GeometricTimingDet* a) {
    const double pi = 3.141592653592;
    double phi = a->phi();
    return( phi>= 0 ? phi:phi+2*pi);   
  }
  
  static double extractPhiModule(const GeometricTimingDet* a) {
    const double pi = 3.141592653592;
    std::vector<const GeometricTimingDet*> const & comp = a->components().back()->components();
    float phi = 0.;
    bool sum = true;
    
    for(auto i : comp){
  if(fabs(i->phi())>pi/2.) { 
    sum = false;
    break;
  }
    }
    
    if(sum){
  for(auto i : comp){
    phi+= i->phi();
  }

  double temp = phi/float(comp.size()) < 0. ? 
    2*pi + phi/float(comp.size()):
    phi/float(comp.size());
  return temp;

    }else{
  for(auto i : comp){
    double phi1 = i->phi() >= 0 ? i->phi(): 
      i->phi()+2*pi; 
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
  
  static double extractPhiGluedModule(const GeometricTimingDet* a) {
    const double pi = 3.141592653592;
    std::vector<const GeometricTimingDet*> comp;
    a->deepComponents(comp);
    float phi = 0.;
    bool sum = true;
    
    for(auto & i : comp){
  if(fabs(i->phi())>pi/2.) {
    sum = false;
    break;
  }
    }
    
    if(sum){
  for(auto & i : comp){
    phi+= i->phi();
  }

  double temp = phi/float(comp.size()) < 0. ? 
    2*pi + phi/float(comp.size()):
    phi/float(comp.size());
  return temp;

    }else{
  for(auto & i : comp){
    double phi1 = i->phi() >= 0 ? i->phi(): 
      i->translation().phi()+2*pi; 
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

  
  static double extractPhiMirror(const GeometricTimingDet* a) {
    const double pi = 3.141592653592;
    double phi = a->phi();
    phi = (phi>= 0 ? phi : phi+2*pi); // (-pi,pi] --> [0,2pi)
    return ( (pi-phi) >= 0 ? (pi-phi) : (pi-phi)+2*pi ); // (-pi,pi] --> [0,2pi)
  }
  
  static double extractPhiModuleMirror(const GeometricTimingDet* a) {
    const double pi = 3.141592653592;
    double phi = extractPhiModule(a); // [0,2pi)
    phi = ( phi <= pi ? phi : phi-2*pi );   // (-pi,pi]   
    return (pi-phi);
  }
  
  static double extractPhiGluedModuleMirror(const GeometricTimingDet* a) {
    const double pi = 3.141592653592;
    double phi = extractPhiGluedModule(a); // [0,2pi)
    phi = ( phi <= pi ? phi : phi-2*pi );   // (-pi,pi]   
    return (pi-phi);
  }
  
  static bool LessR_module(const GeometricTimingDet* a, const GeometricTimingDet* b)
  {
    return a->deepComponents().front()->rho() < b->deepComponents().front()->rho();      
  }
  
  static bool LessR(const GeometricTimingDet* a, const GeometricTimingDet* b)
  {
    return a->rho() < b->rho(); 
  }
  
  
 private:
  virtual void buildComponent(DDFilteredView& , GeometricTimingDet*, std::string) = 0;
protected:
  CmsMTDStringToEnum theCmsMTDStringToEnum;
private:
  virtual void sortNS(DDFilteredView& , GeometricTimingDet*){}
  CmsMTDStringToEnum _CmsMTDStringToEnum;
};

#endif
