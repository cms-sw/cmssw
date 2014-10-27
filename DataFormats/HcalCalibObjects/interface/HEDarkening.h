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
// scenario = 0 - full replacement of HE scintillators
// scenario = 1 - no replacement, full stage darkening
// scenario = 2 - only replace scintillators with expected light yield < 20% for 500 fb-1
// scenario = 3 - replace complete megatiles only in the front 4 layers
//

class HEDarkening {

public:
  HEDarkening(unsigned int scenario = 3);
  ~HEDarkening();

  float degradation(float intlumi, int ieta, int lay) const;

  static const char* scenarioDescription (unsigned int scenario);

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
