#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/HcaluLUTTPGCoder.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream.h>
#include <fstream.h>


HcaluLUTTPGCoder::HcaluLUTTPGCoder(double LSB_GeV) {
  lsbGeV_=LSB_GeV;
  gain_=-1.0;
  pedestal_=0;
  service_=0;
  perpIeta_.reserve(42);
  for (int i=0; i<42; i++) perpIeta_.push_back(1.0);
  //LUT_ = {0};
  //userLUT = false;
  //fillLT();
  // cout<<"************************************TPGcnewCoder setting ***********************************"<<endl;
     
}


void HcaluLUTTPGCoder::setupLUT()
{fillLUT();}

void HcaluLUTTPGCoder::fillLUT(){
  ifstream userfile;
  userfile.open("/uscms/home/mlw/TestTrigPrim/CMSSW_1_0_0_pre1/src/HcalTPGInputLut.doc");
   if( userfile )
    {
      userLUT = true;
      for(int i = 0; i < 256; i++)
	{
	  for(int j = 0; j < 5; j++)
	    {userfile >> LUT_[i][j];}
	}
    }
   userfile.close();
}



void HcaluLUTTPGCoder::setupForChannel(const HcalCalibrations& calib) {
  determineGainPedestal(calib,gain_,pedestal_);
}

void HcaluLUTTPGCoder::setupForAuto(const HcalDbService* service) {
  service_=service;
}

void HcaluLUTTPGCoder::setupGeometry(const CaloGeometry& geom) {
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

void HcaluLUTTPGCoder::determineGainPedestal(const HcalCalibrations& calib, double& gain, int& pedestal) const {
  gain=(calib.gain(0)+calib.gain(1)+calib.gain(2)+calib.gain(3))/4;
  pedestal=int((calib.pedestal(0)+calib.pedestal(1)+calib.pedestal(2)+calib.pedestal(3))/4+0.5);
}

void HcaluLUTTPGCoder::adc2ET(const HBHEDataFrame& df, IntegerCaloSamples& ics) const{
  //userLUT = false;
  //setupLUT();
  // cout<<"************************************TPGcnewCoder setting ***********************************"<<endl;
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
  
  useLUT(df, cs);

  for (int i=0; i<cs.size(); i++) {
    double resp=(cs[i]-p)*g;
    resp*=perpIeta_[df.id().ietaAbs()];
    resp=resp/lsbGeV_+0.5;
    if (resp<0) ics[i]=0;
    else ics[i]=int(resp);
  }
  
}

void HcaluLUTTPGCoder::adc2ET(const HFDataFrame& df, IntegerCaloSamples& ics)  const{
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

  CaloSamples cs;
  
  coder_.adc2fC(df,cs); // convert to fC
  for (int i=0; i<cs.size(); i++) {
    double resp=(cs[i]-p)*g;
    resp*=perpIeta_[df.id().ietaAbs()];
    ics[i]=int(resp/lsbGeV_);
  }
  
}

void HcaluLUTTPGCoder::useLUT(const HBHEDataFrame& df, CaloSamples& lf) const {
  lf=CaloSamples(df.id(),df.size());
  
  for (int i=0; i<df.size(); i++) 
    {
      int adc_ = df[i].adc();
      int j = 0;
      double ieta = df.id().ietaAbs();
       if(ieta <21 && ieta>0  ){j = 1;}
      if(ieta <27 && ieta>20 ){j = 2;}
      if(ieta <29 && ieta>26 ){j = 3;}
      if(ieta <42 && ieta>28 ){j = 4;}
      
      lf[i]=LUT_[adc_][j];
    }
  lf.setPresamples(df.presamples());
}

void HcaluLUTTPGCoder::useLUT(const HODataFrame& df, CaloSamples& lf) const {
  lf=CaloSamples(df.id(),df.size());
  
  for (int i=0; i<df.size(); i++) 
    {
      int adc_ = df[i].adc();
      int j = 0;
      double ieta = df.id().ietaAbs();
      if(ieta <21 && ieta>0  ){j = 1;}
      if(ieta <27 && ieta>20 ){j = 2;}
      if(ieta <29 && ieta>26 ){j = 3;}
      if(ieta <42 && ieta>28 ){j = 4;}
      
      lf[i]=LUT_[adc_][j];
    }
  lf.setPresamples(df.presamples());
}
void HcaluLUTTPGCoder::useLUT(const HFDataFrame& df, CaloSamples& lf) const {
  lf=CaloSamples(df.id(),df.size());
  
  for (int i=0; i<df.size(); i++) 
    {
      int adc_ = df[i].adc();
      int j = 0;
      double ieta = df.id().ietaAbs();
      if(ieta <21 && ieta>0  ){j = 1;}
      if(ieta <27 && ieta>20 ){j = 2;}
      if(ieta <29 && ieta>26 ){j = 3;}
      if(ieta <42 && ieta>28 ){j = 4;}
      
      
      lf[i]=LUT_[adc_][j];
    }
  lf.setPresamples(df.presamples());
 }
