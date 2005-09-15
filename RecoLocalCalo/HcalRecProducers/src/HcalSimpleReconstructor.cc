using namespace std;
#include "RecoLocalCalo/HcalRecProducers/interface/HcalSimpleReconstructor.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/EDProduct/interface/EDCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"
#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"
#include <iostream>

namespace cms
{

  namespace hcal{ 

 class HRSHappySelector : public edm::Selector {
  public:
    HRSHappySelector() { }
  private:
    virtual bool doMatch(const edm::Provenance& p) const {
      //      cout << p << endl;
      return true;
    }
 };


  HcalSimpleReconstructor::HcalSimpleReconstructor(edm::ParameterSet const& conf):
    coder_(0),
    reco_(conf.getParameter<int>("firstSample"),conf.getParameter<int>("samplesToAdd"))

  {
    coder_=new HcalNominalCoder();
    std::string subd=conf.getParameter<std::string>("Subdetector");
    if (!strcasecmp(subd.c_str(),"HBHE")) {
      subdet_=HcalBarrel;
      produces<HBHERecHitCollection>();
    } else if (!strcasecmp(subd.c_str(),"HO")) {
      subdet_=HcalOuter;
      produces<HORecHitCollection>();
    } else if (!strcasecmp(subd.c_str(),"HF")) {
      subdet_=HcalForward;
      produces<HFRecHitCollection>();
    } else {
      std::cout << "HcalSimpleReconstructor is not associated with a specific subdetector!" << std::endl;
    }       

  }

  HcalSimpleReconstructor::~HcalSimpleReconstructor() {
    delete coder_; // ? always ?
  }

  void HcalSimpleReconstructor::produce(edm::Event& e, const edm::EventSetup&)
  {
    double gain[]={0.20,0.20,0.20,0.20};
    double pedestal[]={4.0,4.0,4.0,4.0};
    HcalCalibrations defaultCalib(gain,pedestal);

    if (subdet_==HcalBarrel || subdet_==HcalEndcap) {
       edm::Handle<cms::HBHEDigiCollection> digi;
       // selector?
       HRSHappySelector s;
       e.get(s, digi);

       // create empty output
       std::auto_ptr<cms::HBHERecHitCollection> rec(new cms::HBHERecHitCollection);
       // run the algorithm
       cms::HBHEDigiCollection::const_iterator i;
       for (i=digi->begin(); i!=digi->end(); i++) 	 
	 rec->push_back(reco_.reconstruct(*i,*coder_,defaultCalib));
       // return result
       e.put(rec);
    } else if (subdet_==HcalOuter) {
      edm::Handle<cms::HODigiCollection> digi;
      // selector?
       HRSHappySelector s;
      e.get(s, digi);
      
      // create empty output
      std::auto_ptr<cms::HORecHitCollection> rec(new cms::HORecHitCollection);
      // run the algorithm
      cms::HODigiCollection::const_iterator i;
      for (i=digi->begin(); i!=digi->end(); i++) 	 
	rec->push_back(reco_.reconstruct(*i,*coder_,defaultCalib));
      // return result
      e.put(rec);    
    } else if (subdet_==HcalForward) {
      edm::Handle<cms::HFDigiCollection> digi;
      // selector?
      HRSHappySelector s;
      e.get(s, digi);
      
      // create empty output
      std::auto_ptr<cms::HFRecHitCollection> rec(new cms::HFRecHitCollection);
      // run the algorithm
      cms::HFDigiCollection::const_iterator i;
      for (i=digi->begin(); i!=digi->end(); i++) 	 
	rec->push_back(reco_.reconstruct(*i,*coder_,defaultCalib));
      // return result
      e.put(rec);     
    }    
  }
}
}
