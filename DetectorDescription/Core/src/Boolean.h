#ifndef DETECTOR_DESCRIPTION_CORE_DDI_BOOLEAN_H
#define DETECTOR_DESCRIPTION_CORE_DDI_BOOLEAN_H

#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "Solid.h"

namespace DDI {
  
  class BooleanSolid : public Solid
  {
  public:
    BooleanSolid(const DDSolid & A, const DDSolid & B, 
		 const DDTranslation & t,
		 const DDRotation & r,
		 DDSolidShape s);
    
    const DDSolid & a() const { return a_; }
    const DDSolid & b() const { return b_; }
    const DDTranslation & t() const { return t_; }
    const DDRotation & r() const { return r_; }
    
  protected:
    DDSolid a_, b_;
    DDTranslation t_;
    DDRotation r_;
  };

  class MultiUnion : public Solid
  {
  public:
    MultiUnion(const std::vector<DDSolid> & a,
	       const std::vector<DDTranslation> & t,
	       const std::vector<DDRotation> & r);
    
    const std::vector<DDSolid> & solids() const { return a_; }
    const std::vector<DDTranslation> & t() const { return t_; }
    const std::vector<DDRotation> & r() const { return r_; }
    
  protected:
    std::vector<DDSolid> a_;
    std::vector<DDTranslation> t_;
    std::vector<DDRotation> r_;
  };

  class Union : public BooleanSolid
  {
  public:
    Union(const DDSolid & A, const DDSolid & B,
          const DDTranslation & t,
	  const DDRotation & r);
  };
  
  class Intersection : public BooleanSolid
  {
  public:
    Intersection(const DDSolid & A, const DDSolid & B,
                 const DDTranslation & t,
		 const DDRotation & r);
  };
  
  class Subtraction : public BooleanSolid
  {
  public:
    Subtraction(const DDSolid & A, const DDSolid & B,
                const DDTranslation & t, 
		const DDRotation & r);
  };		    
}
#endif // DETECTOR_DESCRIPTION_CORE_DDI_BOOLEAN_H
