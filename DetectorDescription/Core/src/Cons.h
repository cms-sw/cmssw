#ifndef DDI_Cons_h
#define DDI_Cons_h

#include <iosfwd>
#include "Solid.h"

namespace DDI {
 
  class Cons : public DDI::Solid
  {
  public:
    Cons(double zhalf,
	 double rInMinusZ,
	 double rOutMinusZ,
	 double rInPlusZ,
	 double rOutPlusZ,
	 double startPhi,
	 double deltaPhi);
   
    double volume() const ;
    void stream(std::ostream &) const;	 
  };

}

#endif
