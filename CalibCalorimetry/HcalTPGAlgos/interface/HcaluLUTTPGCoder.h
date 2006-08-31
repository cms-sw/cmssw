#ifndef CALIBCALORIMETRY_HCALTPGALGOS_HCALNOMINALTPGCODER_H
#define CALIBCALORIMETRY_HCALTPGALGOS_HCALNOMINALTPGCODER_H 1

#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"
#include <iostream.h>
#include <fstream.h>
class HcalDbService;
class CaloGeometry;

/** \class HcalNominalTPGCoder
  *  
  * The nominal coder uses the HcalNominalCoder to perform ADC->fC,
  * then uses the averaged gain and pedestal values to convert to GeV (Energy)
  * and the ideal geometry to apply Energy -> ET.
  *
  * $Date: 2006/04/28 02:32:36 $
  * $Revision: 1.3 $
  * \author J. Mans - Minnesota
  */
class HcaluLUTTPGCoder : public HcalTPGCoder {
public:
  HcaluLUTTPGCoder(double LSB_GeV);
  virtual ~HcaluLUTTPGCoder() {}
  void setupForChannel(const HcalCalibrations& calib);
  void setupForAuto(const HcalDbService* service=0);
  void setupGeometry(const CaloGeometry& geom);
  void fillLUT() ;
  void setupLUT();
  virtual void adc2ET(const HBHEDataFrame& df, IntegerCaloSamples& ics) const ;
  virtual void adc2ET(const HFDataFrame& df, IntegerCaloSamples& ics) const;
  void useLUT(const HBHEDataFrame& df, CaloSamples& lf) const;
  void useLUT(const HODataFrame& df, CaloSamples& lf) const;
  void useLUT(const HFDataFrame& df, CaloSamples& lf) const;
  
  // int LUT_[256][4];
  //  ifstream userfile;
private:
  void determineGainPedestal(const HcalCalibrations& calib, double& gain, int& pedestal) const;
  double lsbGeV_;
  HcalNominalCoder coder_;
  double gain_;
  int pedestal_;
  std::vector<double> perpIeta_;
  int LUT_[256][5];
  //std::vector<double> LUT_;
  // double LUT[256][4];
  //  int test;
  bool userLUT;
  // ifstream userfile;
  const HcalDbService* service_;
};

#endif
