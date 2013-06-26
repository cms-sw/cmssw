//  COCOA class header file
//Id:  ALILine.h
//CAT: Model
//
//   Base class for entries 
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _ALILINE_HH
#define _ALILINE_HH
#include <iostream>
#include <CLHEP/Vector/ThreeVector.h>
class ALIPlane;

class ALILine {

public:
  ALILine(){ };  
  ~ALILine(){ };  
  ALILine( const CLHEP::Hep3Vector& point, const CLHEP::Hep3Vector& direction );  
// Next 1 line was added with 22 Mar 2001 
//  CLHEP::Hep3Vector ALILine( const ALILine& l2, bool notParallel = 0);
//0
  CLHEP::Hep3Vector intersect( const ALILine& l2, bool notParallel = 0); 
  CLHEP::Hep3Vector intersect( const ALIPlane& plane, bool notParallel = 1);
  const CLHEP::Hep3Vector& pt() const {return _point;};
  const CLHEP::Hep3Vector& vec() const {return _direction;};

  friend std::ostream& operator << (std::ostream&, const ALILine& li);

private:
  CLHEP::Hep3Vector _point;
  CLHEP::Hep3Vector _direction;

};

#endif
