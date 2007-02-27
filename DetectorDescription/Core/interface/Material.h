#ifndef DDI_Material_h
#define DDI_Material_h

#include <iostream>
#include <vector>
#include <utility>
#include "DetectorDescription/Core/interface/DDMaterial.h"


namespace DDI {

  class Material
  {
  public:
    Material(){}
    Material(double z, double a, double d)
     : z_(z), a_(a), density_(d) { }
     
    Material(double d) : z_(0), a_(0), density_(d) { }  
    virtual ~Material(){}
    int noOfConsituents() const;
    
    double a() const { return a_; }
    double z() const { return z_; }
    double density() const { return density_; }
    
    double& a(){ return a_; }
    double& z(){ return z_; }
    double& density(){ return density_; }
    
    int addMaterial(const DDMaterial & m, double fm)
     { composites_.push_back(std::make_pair(m,fm));
       return noOfConstituents();
     }
    
    const DDMaterial::FractionV::value_type & constituent(int i) const
     { return composites_[i]; }
     
    DDMaterial::FractionV::value_type & constituent(int i)
     { return composites_[i]; }
     
    int noOfConstituents() const { return composites_.size(); } 
    
  protected:    
    double  z_, a_, density_;
    DDMaterial::FractionV composites_;
  };

}
#endif
