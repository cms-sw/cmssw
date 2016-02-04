// Shape.h
//
/*! \class Shape
 * \brief Abstract Class of shape
 * 
 * last change : $Date: 2009/06/02 12:55:17 $
 * by          : $Author: malcles $
 *
*/

#ifndef Shape_H
#define Shape_H

#include "TObject.h"

class Shape: public TObject
{
public:
  //! return the value of the shape at a given time (given in clock unit)
  virtual double eval(double t) const = 0 ;
  //! return the value of the derivative of the shape at a given time (given in clock unit)
  virtual double derivative(double t) const = 0 ;
  //! return the value of the time of the max of the shape (in clock unit).
  virtual double getTimeOfMax() const = 0 ;
  //! Calling this method fill the shape corresponding to a given channel (tower, crystal and gain) 
  virtual bool fillShapeFor(int tower=0, int crystal=0, int gain=0) = 0 ; 

  ClassDef(Shape,1) // Definition of a general interface to pulse shapes
};

#endif
