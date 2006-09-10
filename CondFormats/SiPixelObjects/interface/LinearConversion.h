#ifndef SiPixelObjects_LinearConversion_H
#define SiPixelObjects_LinearConversion_H

namespace sipixelobjects {

class LinearConversion {

public:
  LinearConversion(int offset =0, int slope =1) : theOffset(offset), theSlope(slope) { }
  int convert( int item) const { return theOffset+theSlope*item; }
  int inverse( int item) const { return (item - theOffset)/theSlope; } 
  int offset() const { return theOffset; }
  int slope() const { return theSlope; }

private:
  int theOffset, theSlope;
};

}
#endif
