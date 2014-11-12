#ifndef HcalCalibObjects_HBDarkening_h
#define HcalCalibObjects_HBDarkening_h
//
// Simple class with parameterizaed function to get darkening attenuation 
// coefficiant for SLHC conditions
// intlumi is integrated luminosity (fb-1),
// layer is HB layer number (from 0 up to 16), NB: 1-17 in HcalTestNumbering
//
// scenario = 0 - full replacement of HB scintillators
// scenario = 1 - no replacement, full stage darkening
//

class HBDarkening {

public:
  HBDarkening(unsigned int scenario);
  ~HBDarkening();

  float degradation(float intlumi, int ieta, int lay) const;

  static const char* scenarioDescription (unsigned int scenario);

  // 2 contsants below are used in CalibCalorimetry/HcalPlugins HERecalibration
  // (1) number of HE ieta bins for darkening   
  static const unsigned int nEtaBins = 16; 
  // (2) max. number of HE scint. layers
  static const unsigned int nScintLayers = 17;

private:
  int ieta_shift;
  float lumiscale[nEtaBins][nScintLayers];

};


#endif // HBDarkening_h
