#include "TkBfield.h"

#include "FWCore/Utilities/interface/Exception.h"

// #include <iostream>
// #include <iomanip>
#include <cmath>
#include<algorithm>

using namespace magfieldparam;

namespace {


  BCylParam<float>  fpar1{4.90541f,17.8768f,2.02355f,0.0210538f,0.000321885f,2.37511f,0.00326725f,2.07656f,1.71879f}; // 2.0T-2G
  BCylParam<float>  fpar2{4.41982f,15.7732f,3.02621f,0.0197814f,0.000515759f,2.43385f,0.00584258f,2.11333f,1.76079f}; // 3.0T-2G
  BCylParam<float>  fpar3{4.30161f,15.2586f,3.51926f,0.0183494f,0.000606773f,2.45110f,0.00709986f,2.12161f,1.77038f}; // 3.5T-2G
  BCylParam<float>  fpar4{4.24326f,15.0201f,3.81492f,0.0178712f,0.000656527f,2.45818f,0.00778695f,2.12500f,1.77436f}; // 3.8T-2G
  BCylParam<float>  fpar5{4.21136f,14.8824f,4.01683f,0.0175932f,0.000695541f,2.45311f,0.00813447f,2.11688f,1.76076f}; // 4.0T-2G
  std::string const flds[] = {"2_0T","3_0T","3_5T","3_8T","4_0T"};
  BCylParam<float> const fpars[]{fpar1,fpar2,fpar3,fpar4,fpar5};

  BCylParam<float> const & findPar(std::string fld) {
    auto f = std::find(flds,flds+5,fld);
    if (f-flds>4)    throw cms::Exception("BadParameters") 
		       << "Undefined key - " // abort!\n";
		       <<"Defined keys are: \"2_0T\" \"3_0T\" \"3_5T\" \"3_8T\" and \"4_0T\"\n";
    return fpars[f-flds];

  }
  
}

TkBfield::TkBfield(std::string fld) : bcyl(findPar(fld)) {

}



void TkBfield::getBrfz(float const  * __restrict__ x, float * __restrict__ Brfz)  const {
  float br; float bz;
  float r2=x[0]*x[0]+x[1]*x[1];
  bcyl(x[0]*x[0]+x[1]*x[1],x[2], br, bz); Brfz[0]=std::sqrt(r2)*br; Brfz[1]=0; Brfz[2]=bz;
}

void TkBfield::getBxyz(float const  * __restrict__ x, float * __restrict__ Bxyz)  const {
  float br; float bz;
  float r2=x[0]*x[0]+x[1]*x[1];
  bcyl(r2, x[2], br,bz);
  Bxyz[0]=br*x[0];
  Bxyz[1]=br*x[1];
  Bxyz[2]=bz;
}

