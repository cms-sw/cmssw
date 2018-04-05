#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <unordered_set>
#include <string>

//helper
namespace raw_impl {
  template <class T>
  void get(edm::EDGetTokenT<T> tok, edm::Event& e, const std::string& productName="")
  {
    edm::Handle<T> h_coll;
    e.getByToken(tok, h_coll);
    auto o_coll = std::make_unique<T>();
    if(h_coll.isValid()){
      //copy constructor
      o_coll = std::make_unique<T>(*(h_coll.product()));
    }
    if(!productName.empty()) e.put(std::move(o_coll),productName);
    else e.put(std::move(o_coll));
  }
}

class HcalRawToDigiFake : public edm::global::EDProducer<>
{
public:
  explicit HcalRawToDigiFake(const edm::ParameterSet& ps);
  ~HcalRawToDigiFake() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::StreamID id, edm::Event& e, const edm::EventSetup& c) const override;
private:
  //members
  edm::EDGetTokenT<QIE10DigiCollection> tok_QIE10DigiCollection_;
  edm::EDGetTokenT<QIE11DigiCollection> tok_QIE11DigiCollection_;
  edm::EDGetTokenT<HBHEDigiCollection> tok_HBHEDigiCollection_;
  edm::EDGetTokenT<HFDigiCollection> tok_HFDigiCollection_;
  edm::EDGetTokenT<HODigiCollection> tok_HODigiCollection_;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> tok_TPDigiCollection_;
  edm::EDGetTokenT<HOTrigPrimDigiCollection> tok_HOTPDigiCollection_;
  edm::EDGetTokenT<HcalCalibDigiCollection> tok_CalibDigiCollection_;
  edm::EDGetTokenT<ZDCDigiCollection> tok_ZDCDigiCollection_;
  edm::EDGetTokenT<QIE10DigiCollection> tok_ZDCQIE10DigiCollection_;
  edm::EDGetTokenT<HcalTTPDigiCollection> tok_TTPDigiCollection_;
  const bool unpackCalib_, unpackZDC_, unpackTTP_;
};


HcalRawToDigiFake::HcalRawToDigiFake(edm::ParameterSet const& conf):
  tok_QIE10DigiCollection_   (consumes<QIE10DigiCollection       >(conf.getParameter<edm::InputTag>("QIE10"))),
  tok_QIE11DigiCollection_   (consumes<QIE11DigiCollection       >(conf.getParameter<edm::InputTag>("QIE11"))),
  tok_HBHEDigiCollection_    (consumes<HBHEDigiCollection        >(conf.getParameter<edm::InputTag>("HBHE"))),
  tok_HFDigiCollection_      (consumes<HFDigiCollection          >(conf.getParameter<edm::InputTag>("HF"))),
  tok_HODigiCollection_      (consumes<HODigiCollection          >(conf.getParameter<edm::InputTag>("HO"))),
  tok_TPDigiCollection_      (consumes<HcalTrigPrimDigiCollection>(conf.getParameter<edm::InputTag>("TRIG"))),
  tok_HOTPDigiCollection_    (consumes<HOTrigPrimDigiCollection  >(conf.getParameter<edm::InputTag>("HOTP"))),
  tok_CalibDigiCollection_   (consumes<HcalCalibDigiCollection   >(conf.getParameter<edm::InputTag>("CALIB"))),
  tok_ZDCDigiCollection_     (consumes<ZDCDigiCollection         >(conf.getParameter<edm::InputTag>("ZDC"))),
  tok_ZDCQIE10DigiCollection_(consumes<QIE10DigiCollection       >(conf.getParameter<edm::InputTag>("ZDCQIE10"))),
  tok_TTPDigiCollection_     (consumes<HcalTTPDigiCollection     >(conf.getParameter<edm::InputTag>("TTP"))),
  unpackCalib_(conf.getParameter<bool>("UnpackCalib")),
  unpackZDC_(conf.getParameter<bool>("UnpackZDC")),
  unpackTTP_(conf.getParameter<bool>("UnpackTTP"))
{
  // products produced...
  produces<QIE10DigiCollection>();
  produces<QIE11DigiCollection>();
  produces<HBHEDigiCollection>();
  produces<HFDigiCollection>();
  produces<HODigiCollection>();
  produces<HcalTrigPrimDigiCollection>();
  produces<HOTrigPrimDigiCollection>();
  if (unpackCalib_)
    produces<HcalCalibDigiCollection>();
  if (unpackZDC_)
    produces<ZDCDigiCollection>();
  if (unpackTTP_)
    produces<HcalTTPDigiCollection>();
  produces<QIE10DigiCollection>("ZDC");
}

// Virtual destructor needed.
HcalRawToDigiFake::~HcalRawToDigiFake() { }  

void HcalRawToDigiFake::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("UnpackZDC",true);
  desc.add<bool>("UnpackCalib",true);
  desc.add<bool>("UnpackTTP",true);
  //empty tag = not usually produced by simulation
  desc.add<edm::InputTag>("QIE10", edm::InputTag("simHcalDigis", "HFQIE10DigiCollection"));
  desc.add<edm::InputTag>("QIE11", edm::InputTag("simHcalDigis", "HBHEQIE11DigiCollection"));
  desc.add<edm::InputTag>("HBHE", edm::InputTag("simHcalDigis"));
  desc.add<edm::InputTag>("HF", edm::InputTag("simHcalDigis"));
  desc.add<edm::InputTag>("HO", edm::InputTag("simHcalDigis"));
  desc.add<edm::InputTag>("TRIG", edm::InputTag("simHcalTriggerPrimitiveDigis"));
  desc.add<edm::InputTag>("HOTP", edm::InputTag(""));
  desc.add<edm::InputTag>("CALIB", edm::InputTag(""));
  desc.add<edm::InputTag>("ZDC", edm::InputTag("simHcalUnsuppressedDigis"));
  desc.add<edm::InputTag>("ZDCQIE10", edm::InputTag(""));
  desc.add<edm::InputTag>("TTP", edm::InputTag(""));
  //not used, just for compatibility
  desc.add<edm::InputTag>("InputLabel",edm::InputTag("rawDataCollector"));
  desc.add<int>("firstSample",0);
  desc.add<int>("lastSample",0);
  descriptions.add("HcalRawToDigiFake",desc);
}


// Functions that gets called by framework every event
void HcalRawToDigiFake::produce(edm::StreamID id, edm::Event& e, const edm::EventSetup& es) const
{
  //handle each collection
  raw_impl::get(tok_QIE10DigiCollection_,e);
  raw_impl::get(tok_QIE11DigiCollection_,e);
  raw_impl::get(tok_HBHEDigiCollection_,e);
  raw_impl::get(tok_HFDigiCollection_,e);
  raw_impl::get(tok_HODigiCollection_,e);
  raw_impl::get(tok_TPDigiCollection_,e);
  raw_impl::get(tok_HOTPDigiCollection_,e);
  if(unpackCalib_) raw_impl::get(tok_CalibDigiCollection_,e);
  if(unpackZDC_) raw_impl::get(tok_ZDCDigiCollection_,e);
  raw_impl::get(tok_ZDCQIE10DigiCollection_,e,"ZDC");
  if(unpackTTP_) raw_impl::get(tok_TTPDigiCollection_,e);
}

DEFINE_FWK_MODULE(HcalRawToDigiFake);
