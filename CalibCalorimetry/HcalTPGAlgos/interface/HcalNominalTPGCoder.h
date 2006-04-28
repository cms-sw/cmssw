#ifndef CALIBCALORIMETRY_HCALTPGALGOS_HCALNOMINALTPGCODER_H
#define CALIBCALORIMETRY_HCALTPGALGOS_HCALNOMINALTPGCODER_H 1

#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"

class HcalDbService;
class CaloGeometry;

/** \class HcalNominalTPGCoder
  *  
  * The nominal coder uses the HcalNominalCoder to perform ADC->fC,
  * then uses the averaged gain and pedestal values to convert to GeV (Energy)
  * and the ideal geometry to apply Energy -> ET.
  *
  * $Date: 2006/04/03 16:37:39 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class HcalNominalTPGCoder : public HcalTPGCoder {
public:
  HcalNominalTPGCoder(double LSB_GeV);
  virtual ~HcalNominalTPGCoder() {}
  void setupForChannel(const HcalCalibrations& calib);
  void setupForAuto(HcalDbService* service=0);
  void setupGeometry(const CaloGeometry& geom);
  virtual void adc2ET(const HBHEDataFrame& df, IntegerCaloSamples& ics) const;
  virtual void adc2ET(const HFDataFrame& df, IntegerCaloSamples& ics) const;  
private:
  void determineGainPedestal(const HcalCalibrations& calib, double& gain, int& pedestal) const;
  double lsbGeV_;
  HcalNominalCoder coder_;
  double gain_;
  int pedestal_;
  std::vector<double> perpIeta_;
  HcalDbService* service_;
};

#endif
