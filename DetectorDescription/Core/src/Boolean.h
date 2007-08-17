#ifndef DDI_Boolean_h
#define DDI_Boolean_h
#include "Solid.h"

#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDTransform.h"

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
    
    //double volume() const=0;
  protected:
    DDSolid a_, b_;
    DDTranslation t_;
    DDRotation r_;
  };
  
  class Union : public BooleanSolid
  {
  public:
    Union(const DDSolid & A, const DDSolid & B,
          const DDTranslation & t,
	  const DDRotation & r);
    
    //double volume() const;
  };
  
  class Intersection : public BooleanSolid
  {
  public:
    Intersection(const DDSolid & A, const DDSolid & B,
                 const DDTranslation & t,
		 const DDRotation & r);
    
    //double volume() const;
  };
  
  
  class Subtraction : public BooleanSolid
  {
  public:
    Subtraction(const DDSolid & A, const DDSolid & B,
                const DDTranslation & t, 
		const DDRotation & r);
  };		    
}

#endif // DDI_Boolean_h
