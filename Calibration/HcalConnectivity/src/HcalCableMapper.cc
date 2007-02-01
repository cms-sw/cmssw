
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/Framework/interface/Selector.h"
#include <iostream>

using namespace std;

/*$Date: 2007/01/18 $
 author Kevin Klapoetke - Minnesota*/





class HcalCableMapper : public edm::EDAnalyzer {
public:
  explicit HcalCableMapper(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& c);

  //std::string sourceDigi_;

  edm::InputTag hbheLabel_,hoLabel_,hfLabel_;

};


HcalCableMapper::HcalCableMapper(edm::ParameterSet const& conf) :
  hbheLabel_(conf.getParameter<edm::InputTag>("hbheLabel")),
  hoLabel_(conf.getParameter<edm::InputTag>("hoLabel")),
  hfLabel_(conf.getParameter<edm::InputTag>("hfLabel")){

}

static const char* det_names[] = {"Zero","HcalBarrel","HcalEndcap","HcalOuter","HcalForward"};

template <class DigiCollection>
void process(const DigiCollection& digis) {

  for (typename DigiCollection::const_iterator digi=digis.begin(); digi!=digis.end(); digi++) {
    
    int header = ((digi->sample(0).adc())&0x7F);
    int ieta  = ((digi->sample(1).adc())&0x3F);
    int z_ieta = (((digi->sample(1).adc())>>6)&0x1);
    int iphi = ((digi->sample(2).adc())&0x7F);
    int depth = ((digi->sample(3).adc())&0x7);
    int det = (((digi->sample(3).adc())>>3)&0xF);
    int spigot = ((digi->sample(4).adc())&0xF);
    int fiber = (((digi->sample(4).adc())>>4)&0x7);
    int crate  = ((digi->sample(5).adc())&0x1F);
    int fiber_chan= (((digi->sample(5).adc())>>5)&0x3);
    int G_Dcc  = ((digi->sample(6).adc())&0x3F);
    int H_slot  = ((digi->sample(7).adc())&0x1F);
    int TB = (((digi->sample(7).adc())>>5)&0x1);
    int RBX_7 = (((digi->sample(7).adc())>>6)&0x1);
    int RBX = ((digi->sample(8).adc())&0x7F);
    int RM = ((digi->sample(9).adc())&0x3);
    int RM_card= (((digi->sample(9).adc())>>2)&0x3);
    int RM_chan= (((digi->sample(9).adc())>>4)&0x7);
    string eta_sign;
    std::string det_name;
    if (det>4 || det<0) {
      char c[20];
      snprintf(c,20,"Det=%d",det);
      det_name=c;
    } else det_name=det_names[det];

    //if (header = 0x75){
    if (z_ieta==1){
      eta_sign = "+";
    }else{eta_sign = "-";}
    string is_header;
    if (header == 0x75){


      is_header=" Header found";//}else if(ieta+64==0x75){is_header=" DATA SHIFT";} else{is_header=" +No Header+";}

    std::cout <<" Digi ID: " <<digi->id() << is_header<< " ieta: "<< eta_sign << ieta << " iphi: "<< iphi << " Depth: " << depth << " Detector: " << det_name << " Spigot: "<< spigot << " Fiber: " << fiber+1 << " Fiber Channel: "<< fiber_chan << " Crate: " << crate << " Global Dcc: " << G_Dcc << " HTR Slot: " << H_slot << " Top/Bottom: " << TB  << " RBX: " << (RBX_7*128+RBX) << " RM: " << RM+1 << " RM Card: " << RM_card << " RM Channel: " << RM_chan << std::endl;
    }else if (ieta+64==0x75){


      ieta  = ((digi->sample(2).adc())&0x3F);
      z_ieta = (((digi->sample(2).adc())>>6)&0x1);
      iphi = ((digi->sample(3).adc())&0x7F);
      depth = ((digi->sample(4).adc())&0x7);
      det = (((digi->sample(4).adc())>>3)&0xF);
      spigot = ((digi->sample(5).adc())&0xF);
      fiber = (((digi->sample(5).adc())>>4)&0x7);
      crate  = ((digi->sample(6).adc())&0x1F);
      fiber_chan= (((digi->sample(6).adc())>>5)&0x3);
      G_Dcc  = ((digi->sample(7).adc())&0x3F);
      H_slot  = ((digi->sample(8).adc())&0x1F);
      TB = (((digi->sample(8).adc())>>5)&0x1);
      RBX_7 = (((digi->sample(8).adc())>>6)&0x1);
      RBX = ((digi->sample(9).adc())&0x7F);
      
      is_header=" DATA SHIFT";

      std::cout <<" Digi ID: " <<digi->id() << is_header<< " ieta: "<< eta_sign << ieta << " iphi: "<< iphi << " Depth: " << depth << " Detector: " << det_name << " Spigot: "<< spigot << " Fiber: " << fiber+1 << " Fiber Channel: "<< fiber_chan << " Crate: " << crate << " Global Dcc: " << G_Dcc << " HTR Slot: " << H_slot << " Top/Bottom: " << TB  << " RBX: " << (RBX_7*128+RBX) << std::endl;
    }else { std::cout<<" Digi ID: " <<digi->id() << " +NO HEADER+  " << " RBX: " << (RBX_7*128+RBX) << std::endl;
    }
    
  }
}


void HcalCableMapper::analyze(edm::Event const& e, edm::EventSetup const& c) {
  
  
  
  edm::Handle<HBHEDigiCollection> hbhe;
  e.getByLabel(hbheLabel_,hbhe);

  edm::Handle<HFDigiCollection> hf;
  e.getByLabel(hfLabel_,hf);
  edm::Handle<HODigiCollection> ho;
  e.getByLabel(hoLabel_,ho);
  
  
  
  
  process(*hbhe);
  process(*hf);
  process(*ho);
  
}


#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalCableMapper);

