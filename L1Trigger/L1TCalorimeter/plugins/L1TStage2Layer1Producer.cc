// -*- C++ -*-
//
// Package:    L1Trigger/skeleton
// Class:      skeleton
// 
/**\class skeleton skeleton.cc L1Trigger/skeleton/plugins/skeleton.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  James Brooke
//         Created:  Thu, 05 Dec 2013 17:39:27 GMT
//
//


// system include files
#include <boost/shared_ptr.hpp>

// user include files

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer1FirmwareFactory.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2PreProcessor.h"

#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"

#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"


#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

//
// class declaration
//

using namespace l1t;

  
  class L1TStage2Layer1Producer : public edm::EDProducer { 
  public:
    explicit L1TStage2Layer1Producer(const edm::ParameterSet& ps);
    ~L1TStage2Layer1Producer();
    
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions)
      ;
    
  private:
    virtual void beginJob() override;
    virtual void produce(edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() override;
    
    virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
    virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
    //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    
    // ----------member data ---------------------------

    int verbosity_;
    
    int bxFirst_, bxLast_; // bx range to process
    int ietaMin_, ietaMax_, iphiMin_, iphiMax_;
    
    std::vector<edm::EDGetToken> ecalToken_;  // this is a crazy way to store multi-BX info
    std::vector<edm::EDGetToken> hcalToken_;  // should be replaced with a BXVector< > or similar

    // parameters
    unsigned long long paramsCacheId_;
    unsigned fwv_;
    CaloParams* params_;

    // the processor
    Stage2Layer1FirmwareFactory factory_;
    boost::shared_ptr<Stage2PreProcessor> processor_;
     
  }; 
  


L1TStage2Layer1Producer::L1TStage2Layer1Producer(const edm::ParameterSet& ps) :
  verbosity_(ps.getParameter<int>("verbosity")),
  bxFirst_(ps.getParameter<int>("bxFirst")),
  bxLast_(ps.getParameter<int>("bxLast")),
  ietaMin_(-32),
  ietaMax_(32),
  iphiMin_(1),
  iphiMax_(72),
  ecalToken_(bxLast_+1-bxFirst_),
  hcalToken_(bxLast_+1-bxFirst_),
  paramsCacheId_(0),
  params_(0)
{

  // register what you produce
  produces<CaloTowerBxCollection> ();
  
  // register what you consume and keep token for later access:
  for (int ibx=0; ibx<bxLast_+1-bxFirst_; ibx++) {
    ecalToken_[ibx] = consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("ecalToken"));
    hcalToken_[ibx] = consumes<HcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("hcalToken"));
  }
  
  // placeholder for the parameters
  params_ = new CaloParams;

  // set firmware version from python config for now
  fwv_ = ps.getParameter<int>("firmware");

}

L1TStage2Layer1Producer::~L1TStage2Layer1Producer() {
  
  delete params_;

}

// ------------ method called to produce the data  ------------
void
L1TStage2Layer1Producer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  LogDebug("l1t|stage 2") << "L1TStage2Layer1Producer::produce function called..." << std::endl;

  // do event setup
  // get RCT input scale objects
  edm::ESHandle<L1CaloEcalScale> ecalScale;
  iSetup.get<L1CaloEcalScaleRcd>().get(ecalScale);
  //  const L1CaloEcalScale* e = ecalScale.product();
  
  edm::ESHandle<L1CaloHcalScale> hcalScale;
  iSetup.get<L1CaloHcalScaleRcd>().get(hcalScale);
  //  const L1CaloHcalScale* h = hcalScale.product();

  LogDebug("L1TDebug") << "First BX=" << bxFirst_ << ", last BX=" << bxLast_ << ", LSB(E)=" << params_->towerLsbE() << ", LSB(H)=" << params_->towerLsbH() << std::endl;


  // output collection
  std::auto_ptr<CaloTowerBxCollection> towersColl (new CaloTowerBxCollection);


  // loop over crossings
  for (int bx = bxFirst_; bx < bxLast_+1; ++bx) {
   
    int ibx = bx-bxFirst_;
 
    edm::Handle<EcalTrigPrimDigiCollection> ecalTPs;
    edm::Handle<HcalTrigPrimDigiCollection> hcalTPs;
    
    iEvent.getByToken(hcalToken_[ibx], hcalTPs);
    iEvent.getByToken(ecalToken_[ibx], ecalTPs);

    // create input and output tower vectors for this BX
    std::auto_ptr< std::vector<CaloTower> > localInTowers (new std::vector<CaloTower>(CaloTools::caloTowerHashMax()));
    std::auto_ptr< std::vector<CaloTower> > localOutTowers (new std::vector<CaloTower>()); //this is later filled to the same size as localInTowers
    
    // loop over ECAL TPs
    EcalTrigPrimDigiCollection::const_iterator ecalItr;
    int nEcal=0;
    for (ecalItr=ecalTPs->begin(); ecalItr!=ecalTPs->end(); ++ecalItr, ++nEcal) {

      int ieta = ecalItr->id().ieta();
      int iphi = ecalItr->id().iphi();

      int ietIn = ecalItr->compressedEt();
      bool ifg  = ecalItr->fineGrain();

      // decompress
      double et = ecalScale->et( ietIn, abs(ieta), (ieta>0) );
      int ietOut = floor( et / params_->towerLsbE() );
      //      int ietOutMask = (int) pow(2,params_->towerNBitsE())-1;
      
      if (ietIn>0) 
	LogDebug("L1TDebug") << " ECAL TP : " << ieta << ", " << iphi << ", " << ietIn << ", " << et << ", " << ietOut << std::endl;

      int itow = CaloTools::caloTowerHash(ieta, iphi);
      localInTowers->at(itow).setHwEtEm(ietOut);// & ietOutMask);
      localInTowers->at(itow).setHwQual( localInTowers->at(itow).hwQual() | (ifg ? 0x4 : 0x0) );

    }

    // loop over HCAL TPs
    HcalTrigPrimDigiCollection::const_iterator hcalItr;
    int nHcal=0;
    for (hcalItr=hcalTPs->begin(); hcalItr!=hcalTPs->end(); ++hcalItr, ++nHcal) {
    
      int ieta = hcalItr->id().ieta(); 
      int iphi = hcalItr->id().iphi();

      int ietIn = hcalItr->SOI_compressedEt();
      //int ifg = hcalItr->SOI_fineGrain();

      // decompress
      double et = hcalScale->et( ietIn, abs(ieta), (ieta>0) );
      int ietOut = floor( et / params_->towerLsbH() );
      //      int ietOutMask = (int) pow(2,params_->towerNBitsH() )-1;

      if (ietIn>0) 
	LogDebug("L1TDebug") << " HCAL TP : " << ieta << ", " << iphi << ", " << ietIn << ", " << et << ", " << ietOut << std::endl;

      int itow = CaloTools::caloTowerHash(ieta, iphi);
      localInTowers->at(itow).setHwEtHad(ietOut);// & ietOutMask);
      //      towers.at(itow).setHwFGHad(ifg);

    }

    // now calculate remaining tower quantities
    for (int ieta=ietaMin_; ieta<ietaMax_+1; ieta++) {

      for (int iphi=iphiMin_; iphi<iphiMax_+1; iphi++) {

	if(!CaloTools::isValidIEtaIPhi(ieta,iphi)) continue;

	int itow = CaloTools::caloTowerHash(ieta, iphi);

	// get ECAL/HCAL raw numbers
	int ietEcal = localInTowers->at(itow).hwEtEm();
	int ietHcal = localInTowers->at(itow).hwEtHad();
	
	//	const LorentzVector& p4;
	int iet = ietEcal + ietHcal;   // this is nonsense, temp solution!

	//LogDebug("L1TDebug") << " Tower : " << ieta << ", " << iphi << ", " << iet << ", " << ietEcal << ", " << ietHcal << std::endl;

	localInTowers->at(itow).setHwPt(iet);
	localInTowers->at(itow).setHwEta(ieta);
	localInTowers->at(itow).setHwPhi(iphi);

      }
    }

    // do the decompression
    processor_->processEvent(*localInTowers, *localOutTowers);
    
    // copy towers to output collection
    for(std::vector<CaloTower>::const_iterator tower = localOutTowers->begin(); 
	tower != localOutTowers->end(); 
	++tower) 
      towersColl->push_back(ibx, *tower);

    LogDebug("L1TDebug") << "BX=" << ibx << ", N(Tower in)=" << localInTowers->size() << ", N(Tower out)=" << localOutTowers->size() << std::endl;

  }
 
  iEvent.put(towersColl);
  
}

// ------------ method called once each job just before starting event loop  ------------
void 
L1TStage2Layer1Producer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TStage2Layer1Producer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
L1TStage2Layer1Producer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{

  // update parameters and algorithms at run start, if they have changed
  // update params first because the firmware factory relies on pointer to params

  // parameters

  unsigned long long id = iSetup.get<L1TCaloParamsRcd>().cacheIdentifier();  
  
  if (id != paramsCacheId_) {

    paramsCacheId_ = id;

    edm::ESHandle<CaloParams> paramsHandle;
    iSetup.get<L1TCaloParamsRcd>().get(paramsHandle);

    // replace our local copy of the parameters with a new one using placement new
    params_->~CaloParams();
    params_ = new (params_) CaloParams(*paramsHandle.product());
    
    LogDebug("L1TDebug") << *params_ << std::endl;

    if (! params_){
      edm::LogError("l1t|caloStage2") << "Could not retrieve params from Event Setup" << std::endl;            
    }

  }

  // firmware

  if ( !processor_ ) { // in future, also check if the firmware cache ID has changed !
    
    //     m_fwv = ; // get new firmware version in future
    
    // Set the current algorithm version based on DB pars from database:
    processor_ = factory_.create(fwv_, params_);

    LogDebug("L1TDebug") << "Processor object : " << (processor_?1:0) << std::endl;
        
    if (! processor_) {
      // we complain here once per run
      edm::LogError("l1t|caloStage2") << "Layer 1 firmware could not be configured.\n";
    }
    
  }
  

}


// ------------ method called when ending the processing of a run  ------------
void
L1TStage2Layer1Producer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TStage2Layer1Producer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup cons
t&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TStage2Layer1Producer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&
)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TStage2Layer1Producer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TStage2Layer1Producer);
