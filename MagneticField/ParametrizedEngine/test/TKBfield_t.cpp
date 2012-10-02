#include <MagneticField/ParametrizedEngine/plugins/TkBfield.cc>
#include <iostream>



int main() {
  magfieldparam::TkBfield field; 
  for (double  x=-0.5; x<0.6; x+=0.5)
    for (double y=- 0.5; y<0.6; y+=0.5)
      for (double z=-2.; z<2.1; z+=0.5) {
	double loc[] = {x,y,z};
	double b[3];
	field.getBxyz(loc,b);
	std::cout << b[0] << ", "  << b[1] << ", "  << b[2] << std::endl;	 
      }	       
  return 0;
}	    

