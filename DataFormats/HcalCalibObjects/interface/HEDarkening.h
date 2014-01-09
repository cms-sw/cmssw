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

//  both two contsants are used in CalibCalorimetry/HcalPlugins HERecalibration
#define nEtaBins_HEDarkening     14 // number of HE ieta bins for darkening 
#define nScintLayers_HEDarkening 19 // max. number of HE scint. layers


class HEDarkening {

public:
  HEDarkening();
  ~HEDarkening();

  float degradation(float intlumi, int ieta, int lay);

private:
  int ieta_shift;
  float lumiscale[nEtaBins_HEDarkening][nScintLayers_HEDarkening];

};


#endif // HEDarkening_h
