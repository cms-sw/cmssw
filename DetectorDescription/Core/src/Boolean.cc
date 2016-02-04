#include "DetectorDescription/Core/src/Boolean.h"

DDI::BooleanSolid::BooleanSolid(const DDSolid & A, const DDSolid & B, 
                      const DDTranslation & t,
	              const DDRotation & r,
	              DDSolidShape s)
 : Solid(s), a_(A), b_(B), t_(t), r_(r)
 { }
 
 		      
DDI::Union::Union(const DDSolid & A, const DDSolid & B, 
                  const DDTranslation & t,
	          const DDRotation & r)
 : DDI::BooleanSolid(A,B,t,r,ddunion)
 { }
 

DDI::Intersection::Intersection(const DDSolid & A, const DDSolid & B, 
                  const DDTranslation & t,
	          const DDRotation & r)
 : DDI::BooleanSolid(A,B,t,r,ddintersection)
 { }
 
 
DDI::Subtraction::Subtraction(const DDSolid & A, const DDSolid & B, 
                  const DDTranslation & t,
	          const DDRotation & r)
 : DDI::BooleanSolid(A,B,t,r,ddsubtraction)
 { }
   
      
	          
		      
