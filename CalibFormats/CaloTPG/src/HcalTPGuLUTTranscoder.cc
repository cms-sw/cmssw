#include "CalibFormats/CaloTPG/interface/HcalTPGuLUTTranscoder.h"
#include <iostream.h>
#include <iomanip>
#include <fstream.h>


HcalTPGuLUTTranscoder::HcalTPGuLUTTranscoder(int hcalMeV, int egammaMeV, int jetMeV) : 
  hcalLSBMeV_(hcalMeV),
  egammaLSBMeV_(egammaMeV),
  jetLSBMeV_(jetMeV) {
  if (hcalMeV>egammaMeV) {
    std::cout << "Warning : HCAL LSB is bigger than E-Gamma path LSB!\n";
  }
  buildTable();
  filluLUT();
}


HcalTPGuLUTTranscoder::~HcalTPGuLUTTranscoder() {
}


void HcalTPGuLUTTranscoder::htrCompress(const IntegerCaloSamples& ics, const std::vector<bool>& fineGrain, HcalTriggerPrimitiveDigi& digi) {
  digi=HcalTriggerPrimitiveDigi(ics.id());
  digi.setSize(ics.size());
  digi.setPresamples(ics.presamples());
  HcalTrigTowerDetId detId(ics.id());
  double ieta =detId.ietaAbs();
  for (int i=0; i<ics.size(); i++) {
    int code=ics[i];
    //old simple method
    //if (code<int(hcalTable_.size())) code=hcalTable_[code];
    //read from uLUT
    int j = 0;
    if(ieta <21 && ieta>0  ){j = 1;}
    if(ieta <27 && ieta>20 ){j = 2;}
    if(ieta <29 && ieta>26 ){j = 3;}
    if(ieta <42 && ieta>28 ){j = 4;}
    else code=JET_LUT_SIZE-1;
    code = LUT_[code][j];
    
    digi.setSample(i,HcalTriggerPrimitiveSample(code,fineGrain[i],0,0));
  }
}

void HcalTPGuLUTTranscoder::rctEGammaUncompress(const HcalTriggerPrimitiveDigi& digi, IntegerCaloSamples& ics) {
  ics=IntegerCaloSamples(digi.id(),digi.size());
  for (int i=0; i<digi.size(); i++) {
    unsigned int code=digi[i].compressedEt();
    if (code>egammaTable_.size()) ics[i]=egammaTable_.size()-1;
    else ics[i]=egammaTable_[code];
  }
}

void HcalTPGuLUTTranscoder::rctJetUncompress(const HcalTriggerPrimitiveDigi& digi, IntegerCaloSamples& ics) {
  ics=IntegerCaloSamples(digi.id(),digi.size());
  for (int i=0; i<digi.size(); i++) {
    unsigned int code=digi[i].compressedEt();
    if (code>jetTable_.size()) ics[i]=jetTable_.size()-1;
    else ics[i]=jetTable_[code];
  }
}

void HcalTPGuLUTTranscoder::buildTable() {
  codeMeV_.reserve(JET_LUT_SIZE);

  double ratio=(1.0*egammaLSBMeV_)/(jetLSBMeV_);
  egammaTable_.reserve(EGAMMA_LUT_SIZE);
  jetTable_.reserve(JET_LUT_SIZE);
  for (int i=0; i<EGAMMA_LUT_SIZE; i++) {
    egammaTable_.push_back(i);
    jetTable_.push_back(uint8_t(i*ratio));
    codeMeV_.push_back(i*egammaLSBMeV_);
  }


  int njet=int((JET_LUT_SIZE-EGAMMA_LUT_SIZE)*ratio);
  for (int i=0; i<njet; i++) {
    jetTable_.push_back(std::min(int(jetTable_.back())+1,255));
    codeMeV_.push_back(jetTable_.back()*jetLSBMeV_);
  }
  for (int i=0; i<(JET_LUT_SIZE-EGAMMA_LUT_SIZE-njet); i++) {
    jetTable_.push_back(std::min(int(jetTable_.back())+2,255));
    codeMeV_.push_back(jetTable_.back()*jetLSBMeV_);  
  }

  // now we can construct the HCAL LUT...
  int maxH=codeMeV_.back()/hcalLSBMeV_+1;
  hcalTable_.reserve(maxH);
  int icode=0;    
  for (int i=0; i<maxH; i++) {
    int thisMeV=i*hcalLSBMeV_;
    while (icode<int(codeMeV_.size()-1) && fabs(thisMeV-codeMeV_[icode+1])<fabs(thisMeV-codeMeV_[icode]))
      icode++;
    hcalTable_.push_back(icode);
  }

}

void HcalTPGuLUTTranscoder::printTable() {
  for (int i=0; i<JET_LUT_SIZE; i++) {
    if (i<EGAMMA_LUT_SIZE) {
      std::cout << std::setw(3) << i 
		<< "  " << std::setw(3) << int(egammaTable_[i]) 
		<< "  " << std::setw(3) << int(jetTable_[i]) 
		<< "  " << std::setw(10) << codeMeV_[i];
      bool once=false;
      for (unsigned int j=0; j<hcalTable_.size(); j++)
	if (hcalTable_[j]==i) {
	  if (once) std::cout << ", ";
	  else std::cout << "  ";
	  std::cout << j;
	  once=true;
	}
      std::cout << std::endl;
    } else {
      std::cout << std::setw(3) << i 
		<< "  " << std::setw(3) << ""
		<< "  " << std::setw(3) << int(jetTable_[i]) 
		<< "  " << std::setw(10) << codeMeV_[i];
      bool once=false;
      for (unsigned int j=0; j<hcalTable_.size(); j++)
	if (hcalTable_[j]==i) {
	  if (once) std::cout << ", ";
	  else std::cout << "  ";
	  std::cout << j;
	  once=true;
	}
      std::cout << std::endl;
    }
  }
  
}


void HcalTPGuLUTTranscoder::filluLUT() {
  ifstream userfile;
  userfile.open("/uscms/home/mlw/TestTrigPrim/CMSSW_1_0_0_pre1/src/HcalTPGOutputLut.doc");
  if( userfile )
    {
      for(int i = 0; i < 256; i++)
	{
	  for(int j = 0; j < 5; j++)
	    {userfile >> LUT_[i][j];}
	}
    }
  userfile.close();
}



