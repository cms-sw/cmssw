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

class HEDarkening {

public:
  HEDarkening();
  ~HEDarkening();

  float degradation(float intlumi, int ieta, int lay);

  // 2 contsants below are used in CalibCalorimetry/HcalPlugins HERecalibration
  // (1) number of HE ieta bins for darkening   
  static const unsigned int nEtaBins = 14; 
  // (2) max. number of HE scint. layers
  static const unsigned int nScintLayers = 19;

private:
  int ieta_shift;
  float lumiscale[nEtaBins][nScintLayers];

};


#endif // HEDarkening_h
