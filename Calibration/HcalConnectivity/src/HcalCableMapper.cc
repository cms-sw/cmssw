#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include <iostream>
#include <cstdio>

using namespace std;

/*$Date: 2012/08/05 22:53:25 $
version 3.1 02-13-07 

author Kevin Klapoetke - Minnesota*/

class HcalCableMapper : public edm::EDAnalyzer {
public:
  explicit HcalCableMapper(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  virtual void endJob ();
  //std::string sourceDigi_;
private:
  typedef std::vector<HcalQIESample> SampleSet;
  
  typedef std::map<HcalDetId,SampleSet> PathSet;
  typedef std::map<HcalDetId,HcalElectronicsId> IdMap;
  
  void process(const PathSet& ps, const IdMap& im);
  
  


  std::map<HcalDetId,std::vector<SampleSet> > fullHistory_;
  IdMap IdSet;
  edm::InputTag hbheLabel_,hoLabel_,hfLabel_;

  template <class DigiCollection>
  void record(const DigiCollection& digis) {

    for (typename DigiCollection::const_iterator digi=digis.begin(); digi!=digis.end(); digi++) {

      SampleSet q;
      for (int i=0; i<digi->size(); i++)
	q.push_back(digi->sample(i));
      
      if (fullHistory_.find(digi->id())==fullHistory_.end()) fullHistory_.insert(std::pair<HcalDetId,std::vector<SampleSet> >(digi->id(),std::vector<SampleSet>()));
      if (IdSet.find(digi->id())==IdSet.end()) IdSet.insert(std::pair<HcalDetId,HcalElectronicsId>(digi->id(),HcalElectronicsId())); 
      fullHistory_[digi->id()].push_back(q);
      IdSet[digi->id()]=digi->elecId();
    }
  }


};


HcalCableMapper::HcalCableMapper(edm::ParameterSet const& conf) :
  hbheLabel_(conf.getParameter<edm::InputTag>("hbheLabel")),
  hoLabel_(conf.getParameter<edm::InputTag>("hoLabel")),
  hfLabel_(conf.getParameter<edm::InputTag>("hfLabel")){
  
}

constexpr char const* det_names[] = {"Zero","HcalBarrel","HcalEndcap","HcalForward","HcalOuter"};



void HcalCableMapper::process(const PathSet& ps, const IdMap& im){
  
  PathSet::const_iterator iii;
  IdMap::const_iterator ij;
  
  for (iii=ps.begin();iii!=ps.end();iii++){
    
    
    SampleSet ss = iii->second;
    const HcalDetId dd = iii->first;
    
    ij=im.find(dd);
    HcalElectronicsId eid=ij->second;

    int header = ((ss[0].adc())&0x7F);
    int ieta  = ((ss[1].adc())&0x3F);
    int z_ieta = (((ss[1].adc())>>6)&0x1);
    int iphi = ((ss[2].adc())&0x7F);
    int depth = ((ss[3].adc())&0x7);
    int det = (((ss[3].adc())>>3)&0xF);
    int spigot = ((ss[4].adc())&0xF);
    int fiber = (((ss[4].adc())>>4)&0x7);
    int crate  = ((ss[5].adc())&0x1F);
    int fiber_chan= (((ss[5].adc())>>5)&0x3);
    int G_Dcc  = ((ss[6].adc())&0x3F);
    int H_slot  = ((ss[7].adc())&0x1F);
    int TB = (((ss[7].adc())>>5)&0x1);
    int RBX_7 = (((ss[7].adc())>>6)&0x1);
    int RBX = ((ss[8].adc())&0x7F);
    int RM = ((ss[9].adc())&0x3);
    int RM_card= (((ss[9].adc())>>2)&0x3);
    int RM_chan= (((ss[9].adc())>>4)&0x7);
    string eta_sign;
    std::string det_name;
    if (det>4 || det<0) {
      char c[20];
      snprintf(c,20,"Det=%d",det);
      det_name=c;
    } else det_name=det_names[det];
    
    
    if (z_ieta==1){
      eta_sign = "+";
    }else{eta_sign = "-";}
    string is_header;
    if (header == 0x75){
     
      //NO SHIFT
      if((spigot==eid.spigot())&&(fiber+1==eid.fiberIndex())&&(fiber_chan==eid.fiberChanId())&&(H_slot==eid.htrSlot())&&(G_Dcc==eid.dccid())&&(crate==eid.readoutVMECrateId())&&(iphi==dd.iphi())&&(depth==dd.depth())&&(ieta==dd.ietaAbs())&&(TB==eid.htrTopBottom())&&(det==dd.subdet())){//&&(z_ieta==dd.zside())
	std::cout <<"Pathway match"<<std::endl;
      }else{
	
	is_header=" Header found";
	
	std::cout <<" Digi ID: " << dd << is_header<< " ieta: "<< eta_sign << ieta << " iphi: "<< iphi << " Depth: " << depth << " Detector: " << det_name << " Spigot: "<< spigot<<"/"<<eid.spigot() << " Fiber: " << fiber+1<<"/"<<eid.fiberIndex() << " Fiber Channel: "<< fiber_chan <<"/"<<eid.fiberChanId()<< " Crate: " << crate<<"/"<<eid.readoutVMECrateId() << " Global Dcc: " << G_Dcc <<"/"<<eid.dccid() << " HTR Slot: " << H_slot <<"/ " <<eid.htrSlot()<< " Top/Bottom: " << TB<<"/"<< eid.htrTopBottom() << " RBX: " << (RBX_7*128+RBX) << " RM: " << RM+1 << " RM Card: " << RM_card+1 << " RM Channel: " << RM_chan << std::endl;
      }    
    }else if (ieta+64==0x75){
      
      ieta  = ((ss[2].adc())&0x3F);
      z_ieta = (((ss[2].adc())>>6)&0x1);
      iphi = ((ss[3].adc())&0x7F);
      depth = ((ss[4].adc())&0x7);
      det = (((ss[4].adc())>>3)&0xF);
      spigot = ((ss[5].adc())&0xF);
      fiber = (((ss[5].adc())>>4)&0x7);
      crate  = ((ss[6].adc())&0x1F);
      fiber_chan= (((ss[6].adc())>>5)&0x3);
      G_Dcc  = ((ss[7].adc())&0x3F);
      H_slot  = ((ss[8].adc())&0x1F);
      TB = (((ss[8].adc())>>5)&0x1);
      RBX_7 = (((ss[8].adc())>>6)&0x1);
      RBX = ((ss[9].adc())&0x7F);
      

      //SHIFT
      	if((spigot==eid.spigot())&&(fiber+1==eid.fiberIndex())&&(fiber_chan==eid.fiberChanId())&&(H_slot==eid.htrSlot())&&(G_Dcc==eid.dccid())&&(TB==eid.htrTopBottom())&&(crate==eid.readoutVMECrateId())&&(iphi==dd.iphi())&&(depth==dd.depth())&&(det==dd.subdet())&&(ieta==dd.ietaAbs())){//&&(z_ieta==dd.zside())

	std::cout <<"Pathway match (SHIFT)"<<std::endl;
      }else{


      is_header=" DATA SHIFT";
      
     std::cout <<" Digi ID: " << dd << is_header<< " ieta: "<< eta_sign << ieta << " iphi: "<< iphi << " Depth: " << depth << " Detector: " << det_name << " Spigot: "<< spigot<<"/"<<eid.spigot() << " Fiber: " << fiber+1<<"/"<<eid.fiberIndex() << " Fiber Channel: "<< fiber_chan <<"/"<<eid.fiberChanId()<< " Crate: " << crate<<"/"<<eid.readoutVMECrateId() << " Global Dcc: " << G_Dcc <<"/"<<eid.dccid() << " HTR Slot: " << H_slot <<"/ " <<eid.htrSlot()<< " Top/Bottom: " << TB<<"/"<< eid.htrTopBottom() << " RBX: " << (RBX_7*128+RBX) << std::endl;
      
	}
    }else { std::cout<<" Digi ID: " <<dd << " +NO HEADER+  " << " RBX: " << (RBX_7*128+RBX) << std::endl;
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
  
   
  record(*hbhe);
  record(*hf);  
  record(*ho);
}




void HcalCableMapper::endJob(){
  
  
  std::vector<SampleSet>::iterator j; 
  int c [128];
  int k,ii,kk;
  int c_max=0;
  
  std::map<HcalDetId,std::vector<SampleSet> >::iterator i;
  
  PathSet consensus;
 
  for (i=fullHistory_.begin(); i!=fullHistory_.end(); i++) {
    //i.first --> id
    //i.second --> vector<SampleSet>
    SampleSet s;
    for (k=0; k<10; k++) {
      for (ii=0; ii<128; ii++) c[ii]=0;

      for (j=i->second.begin();j!=i->second.end();j++){//word number
	if (int(j->size())>k) 
	  c[(*j)[k].adc()]++;
	
    
      }//j loop
      //sort c-array
      for (kk=0;kk<128;kk++){  
	if (c[kk]>c[c_max]){
	  c_max = kk;
	}
      }    
      
      s.push_back(((c_max&0x7F)));

      c_max=0;
    }//k-loop    
  consensus[i->first]=s;
  
  }//i loop
 
  process(consensus,IdSet);
    
   



}//end of endjob 
  



#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


DEFINE_FWK_MODULE(HcalCableMapper);

