#include <MagneticField/ParametrizedEngine/plugins/TkBfield.cc>
#include <iostream>



int main() {
  magfieldparam::TkBfield field; 
  for (float  x=-0.5; x<0.6; x+=0.5)
    for (float y=- 0.5; y<0.6; y+=0.5)
      for (float z=-2.; z<2.1; z+=0.5) {
	float loc[] = {x,y,z};
	float b[3];
	field.getBxyz(loc,b);
	std::cout << b[0] << ", "  << b[1] << ", "  << b[2] << std::endl;	 
      }	       
  return 0;
}	    

