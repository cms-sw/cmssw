#ifndef HcalCalibObjects_HEDarkening_h
#define HcalCalibObjects_HEDarkening_h
//
// Simple class with parameterizaed function to get darkening attenuation 
// coefficiant for SLHC conditions
// = degradation(int_lumi(intlumi) * dose(layer,Radius)), where
// intlumi is integrated luminosity (fb-1), 
// layer is HE layer number (from -1 up// to 17), NB: 1-19 in HcalTestNumbering
// Radius is radius from the beam line (cm) 
//

#define maxEta 14
#define maxLay 19

class HEDarkening {

public:
  HEDarkening();
  ~HEDarkening();

  float degradation(float intlumi, int ieta, int lay);

private:
  int ieta_shift;
  float lumiscale[maxEta][maxLay];

};


#endif // HEDarkening_h
