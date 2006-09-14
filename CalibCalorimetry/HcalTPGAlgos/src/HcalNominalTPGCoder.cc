#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/HcalNominalTPGCoder.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HcalTowerAlgo/src/HcalHardcodeGeometryData.h"


HcalNominalTPGCoder::HcalNominalTPGCoder(double LSB_GeV, bool doET) {
  lsbGeV_=LSB_GeV;
  gain_=-1.0;
  pedestal_=0;
  service_=0;
  perpIeta_.reserve(42);
  perpIeta_.push_back(1.0); // zero
  // HB+HE
  for (int i=1; i<29; i++) 
    if (doET) perpIeta_.push_back(1.0/cosh((theHBHEEtaBounds[i-1]+theHBHEEtaBounds[i])/2));
    else perpIeta_.push_back(1.0);
  // HF
  for (int i=1; i<13; i++) 
    if (doET) perpIeta_.push_back(1.0/cosh((theHFEtaBounds[i]+theHFEtaBounds[i+1])/2));
    else perpIeta_.push_back(1.0);
}

void HcalNominalTPGCoder::getConditions(const edm::EventSetup& es) const {
  edm::ESHandle<HcalDbService> conditions;
  es.get<HcalDbRecord>().get(conditions);
  service_=conditions.product();
}

void HcalNominalTPGCoder::releaseConditions() const {
  service_=0;
}

void HcalNominalTPGCoder::setupForChannel(const HcalCalibrations& calib) {
  determineGainPedestal(calib,gain_,pedestal_);
}

void HcalNominalTPGCoder::setupForAuto(const HcalDbService* service) {
  service_=service;
}

void HcalNominalTPGCoder::determineGainPedestal(const HcalCalibrations& calib, double& gain, int& pedestal) const {
  gain=(calib.gain(0)+calib.gain(1)+calib.gain(2)+calib.gain(3))/4;
  pedestal=int((calib.pedestal(0)+calib.pedestal(1)+calib.pedestal(2)+calib.pedestal(3))/4+0.5);
}

void HcalNominalTPGCoder::adc2Linear(const HBHEDataFrame& df, IntegerCaloSamples& ics) const {
  double g;
  int p;
  if (service_!=0) {
    HcalCalibrations c;
    if (service_->makeHcalCalibration(df.id(),&c)) 
      determineGainPedestal(c,g,p);
  } else {
    g=gain_;
    p=pedestal_;
  }

  //  ics.setSize(df.size());
  CaloSamples cs;

  coder_.adc2fC(df,cs); // convert to fC
  for (int i=0; i<cs.size(); i++) {
    double resp=(cs[i]-p)*g;
    resp*=perpIeta_[df.id().ietaAbs()];
    resp=resp/lsbGeV_+0.5;
    if (resp<0) ics[i]=0;
    else ics[i]=int(resp);
  }
  
}

void HcalNominalTPGCoder::adc2Linear(const HFDataFrame& df, IntegerCaloSamples& ics) const {
  double g;
  int p;
  if (service_!=0) {
    HcalCalibrations c;
    if (service_->makeHcalCalibration(df.id(),&c)) 
      determineGainPedestal(c,g,p);
  } else {
    g=gain_;
    p=pedestal_;
  }

  //  ics.setSize(df.size());
  CaloSamples cs;

  coder_.adc2fC(df,cs); // convert to fC
  for (int i=0; i<cs.size(); i++) {
    double resp=(cs[i]-p)*g;
    resp*=perpIeta_[df.id().ietaAbs()];
    ics[i]=int(resp/lsbGeV_);
  }
  
}
