#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/HcalNominalTPGCoder.h"
#include "FWCore/Utilities/interface/Exception.h"

HcalNominalTPGCoder::HcalNominalTPGCoder(double LSB_GeV) {
  lsbGeV_=LSB_GeV;
  gain_=-1.0;
  pedestal_=0;
  service_=0;
  perpIeta_.reserve(42);
  for (int i=0; i<42; i++) perpIeta_.push_back(1.0);
}

void HcalNominalTPGCoder::setupForChannel(const HcalCalibrations& calib) {
  determineGainPedestal(calib,gain_,pedestal_);
}

void HcalNominalTPGCoder::setupForAuto(HcalDbService* service) {
  service_=service;
}

void HcalNominalTPGCoder::setupGeometry(const CaloGeometry& geom) {
  int iphi=1, depth=1;
  
  for (int ieta=1; ieta<=41; ieta++) {
    HcalSubdetector sd=(ieta<17)?(HcalBarrel):((ieta<29)?(HcalEndcap):(HcalForward));
    HcalDetId id(sd,ieta,iphi,depth);
    const CaloCellGeometry* cell=geom.getGeometry(id);
    if (cell==0) {
      throw cms::Exception("NullPointer") << "No geometry information for " << id;
    }
    perpIeta_[ieta]=1.0/cosh(cell->getPosition().eta());
  }
}

void HcalNominalTPGCoder::determineGainPedestal(const HcalCalibrations& calib, double& gain, int& pedestal) const {
  gain=(calib.gain(0)+calib.gain(1)+calib.gain(2)+calib.gain(3))/4;
  pedestal=int((calib.pedestal(0)+calib.pedestal(1)+calib.pedestal(2)+calib.pedestal(3))/4+0.5);
}

void HcalNominalTPGCoder::adc2ET(const HBHEDataFrame& df, IntegerCaloSamples& ics) const {
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

void HcalNominalTPGCoder::adc2ET(const HFDataFrame& df, IntegerCaloSamples& ics) const {
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
