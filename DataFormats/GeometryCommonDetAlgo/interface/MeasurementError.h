#ifndef MeasurementError_H
#define MeasurementError_H

/** A very simple class for 2D error matrix components,
 *  used for the measurement frame.
 */

class MeasurementError {
public:

  MeasurementError() : theuu(0), theuv(0), thevv(0) {}

  MeasurementError( float uu, float uv, float vv) :
    theuu(uu), theuv(uv), thevv(vv) {}

  float uu() const { return theuu;}
  float uv() const { return theuv;}
  float vv() const { return thevv;}

private:

  float theuu;
  float theuv;
  float thevv;

};  


#endif // MeasurementError_H
