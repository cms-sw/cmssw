// -*- C++ -*-
//
// Package:    HeavyIonAnalyzer/ZDCAnalysis/ZDCRecHitAnalyzerHC
// Class:      ZDCRecHitAnalyzerHC
//
/**\class ZDCRecHitAnalyzerHC ZDCRecHitAnalyzerHC.cc HeavyIonAnalyzer/ZDCAnalysis/plugins/ZDCRecHitAnalyzerHC

   Description: Produced Tree with ZDC RecHit and zdcdigi information 
*/
//
// Original Author:  Matthew Nickel, University of Kansas
//         Created:  08-10-2024
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TTree.h"

#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

#include "ZDCstruct.h"
#include "ZDCHardCodeHelper.h"

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.

using reco::TrackCollection;

class ZDCRecHitAnalyzerHC : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit ZDCRecHitAnalyzerHC(const edm::ParameterSet&);
  ~ZDCRecHitAnalyzerHC();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<QIE10DigiCollection> ZDCDigiToken_;  
  const edm::EDGetTokenT<edm::SortedCollection<ZDCRecHit>> ZDCRecHitToken_;
  const edm::EDGetTokenT<edm::SortedCollection<ZDCRecHit>> AuxZDCRecHitToken_; 
  edm::ESGetToken<HcalDbService, HcalDbRecord> hcalDatabaseToken_;
  bool doZdcRecHits_;
  bool doZdcDigis_;
  bool doAuxZdcRecHits_;
  bool skipRpdRecHits_;
  bool skipRpdDigis_;
  bool doHardcodedRPD_;
  edm::Service<TFileService> fs;
  TTree *t1, *t2;   
   
  MyZDCDigi zdcDigi;
  MyZDCRecHit zdcRechit;

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  edm::ESGetToken<SetupData, SetupRecord> setupToken_;
#endif
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ZDCRecHitAnalyzerHC::ZDCRecHitAnalyzerHC(const edm::ParameterSet& iConfig) :
  ZDCDigiToken_(consumes<QIE10DigiCollection>(iConfig.getParameter<edm::InputTag>("ZDCDigiSource"))), 
  ZDCRecHitToken_(consumes<edm::SortedCollection<ZDCRecHit>>(iConfig.getParameter<edm::InputTag>("ZDCRecHitSource"))), 
  AuxZDCRecHitToken_(consumes<edm::SortedCollection<ZDCRecHit>>(iConfig.getParameter<edm::InputTag>("AuxZDCRecHitSource"))), 
  hcalDatabaseToken_(esConsumes<HcalDbService, HcalDbRecord>()),
  doZdcRecHits_(iConfig.getParameter<bool>("doZdcRecHits")),
  doZdcDigis_(iConfig.getParameter<bool>("doZdcDigis")),
  doAuxZdcRecHits_(iConfig.getParameter<bool>("doAuxZdcRecHits")),
  skipRpdRecHits_(iConfig.getParameter<bool>("skipRpdRecHits")),
  skipRpdDigis_(iConfig.getParameter<bool>("skipRpdDigis")),
  doHardcodedRPD_(iConfig.getParameter<bool>("doHardcodedRPD"))
{
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  setupDataToken_ = esConsumes<SetupData, SetupRecord>();
#endif
  //now do what ever initialization is needed
}

ZDCRecHitAnalyzerHC::~ZDCRecHitAnalyzerHC() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

// ------------ method called for each event  ------------
void ZDCRecHitAnalyzerHC::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  ZDCHardCodeHelper HardCodeZDC;
  edm::Handle<QIE10DigiCollection> zdcdigis;
  if (doZdcDigis_ || doHardcodedRPD_) {
    iEvent.getByToken(ZDCDigiToken_, zdcdigis);
  }
  edm::ESHandle<HcalDbService> conditions = iSetup.getHandle(hcalDatabaseToken_);  

  if (doZdcRecHits_) {
    zdcRechit.n = 0;
    zdcRechit.sumPlus = 0;
    zdcRechit.sumMinus = 0;
    for (unsigned int i = 0; i < NMOD; i++) {
      zdcRechit.zside[i] = -99;
      zdcRechit.section [i]= -99;
      zdcRechit.channel[i] = -99;
       
      zdcRechit.energy[i] = -99;
      zdcRechit.time[i] = -99;
      zdcRechit.TDCtime[i] = -99;
      zdcRechit.chargeWeightedTime[i] = -99;
      zdcRechit.energySOIp1[i] = -99;
      zdcRechit.ratioSOIp1[i] = -99;
      zdcRechit.saturation[i] = -99;
    }
     
    edm::Handle<ZDCRecHitCollection> zdcrechits;
    iEvent.getByToken(ZDCRecHitToken_, zdcrechits);

    int nhits = 0;
    for (auto const& rh : *zdcrechits) { // does not have RPD if it was skipped in reco
       
      HcalZDCDetId zdcid = rh.id();
      int zside = zdcid.zside();
      int section = zdcid.section();
      int channel = zdcid.channel(); 
      float energy = rh.energy();
       
      if (nhits >= NMOD) break;
      if (section == 1 && channel > 5) continue; // ignore extra EM channels
      if (skipRpdRecHits_ && section == 4) continue;
      
      zdcRechit.zside[nhits] = zside;
      zdcRechit.section[nhits] = section;
      zdcRechit.channel[nhits] = channel;

      // !! Geometry updated for the RPD are not yet part of 14_1_X and the global tag used for 2024. Those won't be available this year but should be next year.
      if (!(doHardcodedRPD_ && section == 4)) {
        zdcRechit.energy[nhits]  = energy;
        zdcRechit.time[nhits]  = rh.time();
        zdcRechit.chargeWeightedTime[nhits] = rh.chargeWeightedTime();
        zdcRechit.energySOIp1[nhits]  = rh.energySOIp1();
        zdcRechit.ratioSOIp1[nhits]  = rh.ratioSOIp1();
        zdcRechit.TDCtime[nhits]  = rh.TDCtime();
        zdcRechit.saturation[nhits] = static_cast<int>( rh.flagField(HcalCaloFlagLabels::ADCSaturationBit) );
      }
      if(zside < 0 && (section == 1 || section == 2)) zdcRechit.sumMinus += energy;
      if(zside > 0 && (section == 1 || section == 2)) zdcRechit.sumPlus += energy;
      
      nhits++;
    } // end loop zdc rechits

    // Fill out RPD module rechits by digi
    // ! Now rely on the same order to match digi and rechit. Ideally matching should be done by zside, section, channel.
    if (zdcdigis.isValid() && !skipRpdRecHits_) {
      nhits = 0;
      for (auto it = zdcdigis->begin(); it != zdcdigis->end(); it++) {
        const QIE10DataFrame digi = static_cast<const QIE10DataFrame>(*it);
        HcalZDCDetId zdcid = digi.id();
        int zside = zdcid.zside();
        int section = zdcid.section();
        int channel = zdcid.channel(); 
      
        if (nhits >= NMOD) break;
        if (section == 1 && channel > 5) continue; // ignore extra EM channels
      
        if (section == 4) { // if RPD was skipped in reco, it wouldn't appear in the previous loop
          zdcRechit.zside[nhits] = zside;
          zdcRechit.section[nhits] = section;
          zdcRechit.channel[nhits] = channel;
        
          if (doHardcodedRPD_) {
            zdcRechit.energy[nhits]  = HardCodeZDC.rechit_Energy_RPD(digi);
            zdcRechit.time[nhits]  = HardCodeZDC.rechit_Time(digi);
            zdcRechit.chargeWeightedTime[nhits] = HardCodeZDC.rechit_ChargeWeightedTime(digi);
            zdcRechit.energySOIp1[nhits]  = HardCodeZDC.rechit_EnergySOIp1(digi);
            zdcRechit.ratioSOIp1[nhits]  = HardCodeZDC.rechit_RatioSOIp1(digi);
            zdcRechit.TDCtime[nhits]  = HardCodeZDC.rechit_TDCtime(digi);
            zdcRechit.saturation[nhits] = HardCodeZDC.rechit_Saturation(digi);
          } // if (!skipRpdRecHits_ && doHardcodedRPD_) {
        } // if (section == 4) {
        nhits++;
      } // for (auto it = zdcdigis->begin(); it != zdcdigis->end(); it++) {
    } // if (zdcdigis.isValid()) {

    zdcRechit.n = nhits;
  } // if(doZdcRecHits_) {

  if (doAuxZdcRecHits_) {
    zdcRechit.sumPlus_Aux = 0;
    zdcRechit.sumMinus_Aux = 0;
     
    edm::Handle<ZDCRecHitCollection> auxrechits;
    iEvent.getByToken(AuxZDCRecHitToken_, auxrechits);
    for (auto const& rh : *auxrechits) {

      HcalZDCDetId zdcid = rh.id();
      int zside = zdcid.zside();
      int section = zdcid.section();
      int channel = zdcid.channel(); 
      float energy = rh.energy();      

      if (section == 1 && channel > 5) continue; // ignore extra EM channels
      
      if (zside <0 && (section ==1 || section ==2)) zdcRechit.sumMinus_Aux += energy;
      if (zside >0 && (section ==1 || section ==2)) zdcRechit.sumPlus_Aux += energy;
      
    } // end loop Aux rechits 

  } // if (doAuxZdcRecHits_) {

  if (t1) t1->Fill();
  
  if (doZdcDigis_) {
    zdcDigi.n = 0;
    for (unsigned int i = 0; i < NMOD; i++) {
      zdcDigi.zside[i] = -99;
      zdcDigi.section[i]= -99;
      zdcDigi.channel[i] = -99;
      for (int ts = 0; ts < NTS; ts++) {
        zdcDigi.chargefC[ts][i] = -99;
        zdcDigi.adc[ts][i] = -99;
        zdcDigi.tdc[ts][i] = -99;
      }
    }
    
    int nhits = 0;
    for (auto it = zdcdigis->begin(); it != zdcdigis->end(); it++) {      
      const QIE10DataFrame digi = static_cast<const QIE10DataFrame>(*it);
      
      HcalZDCDetId zdcid = digi.id();
      int zside = zdcid.zside();
      int section = zdcid.section();
      int channel = zdcid.channel();

      if (nhits >= NMOD) break;
      if (section == 1 && channel > 5) continue; // ignore extra EM channels
      if (skipRpdDigis_ && section == 4) continue;      
      
      CaloSamples caldigi;
      
      //const ZDCDataFrame & rh = (*zdcdigis)[it];

      if (!(doHardcodedRPD_ && section == 4)) {
        const HcalQIECoder* qiecoder = conditions->getHcalCoder(zdcid);
        const HcalQIEShape* qieshape = conditions->getHcalShape(qiecoder);
        HcalCoderDb coder(*qiecoder, *qieshape);
        // coder.adc2fC(rh,caldigi);
        coder.adc2fC(digi, caldigi);
      }
      
      zdcDigi.zside[nhits] = zside;
      zdcDigi.section[nhits] = section;
      zdcDigi.channel[nhits] = channel;
      
      for (int ts = 0; ts < digi.samples(); ts++) {
        zdcDigi.adc[ts][nhits] = digi[ts].adc();
        zdcDigi.tdc[ts][nhits] = digi[ts].le_tdc();
        if (doHardcodedRPD_ && section == 4) {
          zdcDigi.chargefC[ts][nhits] = HardCodeZDC.charge(digi[ts].adc(),digi[ts].capid());
        } else {
          zdcDigi.chargefC[ts][nhits] = caldigi[ts];
        }
      }
      
      nhits++;
    }  // end loop zdc digis

    zdcDigi.n = nhits;

    t2->Fill();
  } // if(doZdcDigis_)
  

  // #ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  // // if the SetupData is always needed
  // auto setup = iSetup.getData(setupToken_);
  // // if need the ESHandle to check if the SetupData was there or not
  // auto pSetup = iSetup.getHandle(setupToken_);
  // #endif                                       
}

// ------------ method called once each job just before starting event loop  ------------
void ZDCRecHitAnalyzerHC::beginJob() {
  // please remove this method if not needed
  
  if(doZdcRecHits_) {
    t1 = fs->make<TTree>("zdcrechit", "zdcrechit");
     
    t1->Branch("sumPlus", &zdcRechit.sumPlus); 
    t1->Branch("sumMinus", &zdcRechit.sumMinus);
     
    t1->Branch("n", &zdcRechit.n);
    t1->Branch("zside", zdcRechit.zside, "zside[n]/I");
    t1->Branch("section", zdcRechit.section, "section[n]/I");
    t1->Branch("channel", zdcRechit.channel, "channel[n]/I");
    t1->Branch("energy", zdcRechit.energy, "energy[n]/F");    
    t1->Branch("time", zdcRechit.time, "time[n]/F");    
    t1->Branch("TDCtime", zdcRechit.TDCtime, "TDCtime[n]/F");    
    t1->Branch("chargeWeightedTime", zdcRechit.chargeWeightedTime, "chargeWeightedTime[n]/F");    
    t1->Branch("energySOIp1", zdcRechit.energySOIp1, "energySOIp1[n]/F");    
    t1->Branch("ratioSOIp1", zdcRechit.ratioSOIp1, "ratioSOIp1[n]/F");    
    t1->Branch("saturation", zdcRechit.saturation, "saturation[n]/I");     
  }
  if(doAuxZdcRecHits_) {
    if (!t1)
      t1 = fs->make<TTree>("zdcrechit", "zdcrechit");
    
    t1->Branch("sumPlus_Aux",&zdcRechit.sumPlus_Aux); 
    t1->Branch("sumMinus_Aux",&zdcRechit.sumMinus_Aux);     
  }
  
  if(doZdcDigis_) {
    t2 = fs->make<TTree>("zdcdigi", "zdcdigi");
    
    t2->Branch("n", &zdcDigi.n, "n/I");
    t2->Branch("zside", zdcDigi.zside, "zside[n]/I");
    t2->Branch("section", zdcDigi.section, "section[n]/I");
    t2->Branch("channel", zdcDigi.channel, "channel[n]/I");

    for (int i = 0; i < NTS; i++) {
      TString adcTsSt("adcTs"), chargefCTsSt("chargefCTs"), tdcTsSt("tdcTs");
      adcTsSt += i;
      chargefCTsSt += i;
      tdcTsSt += i;

      t2->Branch(adcTsSt, zdcDigi.adc[i], adcTsSt + "[n]/I");
      t2->Branch(chargefCTsSt, zdcDigi.chargefC[i], chargefCTsSt + "[n]/F");
      t2->Branch(tdcTsSt, zdcDigi.tdc[i], tdcTsSt + "[n]/I");
    }
  }
    
}

// ------------ method called once each job just after ending the event loop  ------------
void ZDCRecHitAnalyzerHC::endJob() {
  // please remove this method if not needed
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void ZDCRecHitAnalyzerHC::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ZDCRecHitAnalyzerHC);
