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

#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"


//
// class declaration
//

namespace l1t {
  
  class L1TCaloTowerProducer : public edm::EDProducer { 
  public:
    explicit L1TCaloTowerProducer(const edm::ParameterSet& ps);
    ~L1TCaloTowerProducer();
    
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
    CaloParams* params_;
    
  }; 
  
} 


l1t::L1TCaloTowerProducer::L1TCaloTowerProducer(const edm::ParameterSet& ps) :
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
  produces< BXVector<l1t::CaloTower> > ();
  
  // register what you consume and keep token for later access:
  for (int ibx=0; ibx<bxLast_+1-bxFirst_; ibx++) {
    ecalToken_[ibx] = consumes<EcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("ecalToken"));
    hcalToken_[ibx] = consumes<HcalTrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("hcalToken"));
  }

  //  LogDebug("L1TDebug") << "Got here" << std::endl;

  params_ = new CaloParams;

  // params_->setTowerLsbH(0.5);
  // params_->setTowerLsbE(0.5);
  // params_->setTowerNBitsH(8);
  // params_->setTowerNBitsE(8);

}

l1t::L1TCaloTowerProducer::~L1TCaloTowerProducer() {

}



// ------------ method called to produce the data  ------------
void
l1t::L1TCaloTowerProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  std::auto_ptr<l1t::CaloTowerBxCollection> towersColl (new l1t::CaloTowerBxCollection);


  // do event setup
  // using RCT input scale objects
  edm::ESHandle<L1CaloEcalScale> ecalScale;
  iSetup.get<L1CaloEcalScaleRcd>().get(ecalScale);
  //  const L1CaloEcalScale* e = ecalScale.product();
  
  edm::ESHandle<L1CaloHcalScale> hcalScale;
  iSetup.get<L1CaloHcalScaleRcd>().get(hcalScale);
  //  const L1CaloHcalScale* h = hcalScale.product();

  LogDebug("L1TDebug") << "First BX=" << bxFirst_ << ", last BX=" << bxLast_ << ", LSB(E)=" << params_->towerLsbE() << ", LSB(H)=" << params_->towerLsbH() << std::endl;

  // loop over crossings
  for (int bx = bxFirst_; bx < bxLast_+1; ++bx) {
   
    int ibx = bx-bxFirst_;
 
    edm::Handle<EcalTrigPrimDigiCollection> ecalTPs;
    edm::Handle<HcalTrigPrimDigiCollection> hcalTPs;
    
    iEvent.getByToken(hcalToken_[ibx], hcalTPs);
    iEvent.getByToken(ecalToken_[ibx], ecalTPs);

    // create output vector
    std::vector< l1t::CaloTower > towers( CaloTools::caloTowerHashMax() );

    // loop over ECAL TPs
    EcalTrigPrimDigiCollection::const_iterator ecalItr;
    int nEcal=0;
    for (ecalItr=ecalTPs->begin(); ecalItr!=ecalTPs->end(); ++ecalItr, ++nEcal) {

      int ieta = ecalItr->id().ieta();
      int iphi = ecalItr->id().iphi();

      int ietIn = ecalItr->compressedEt();
      //int ifg = ecalItr->fineGrain();

      // decompress
      double et = ecalScale->et( ietIn, abs(ieta), (ieta>0) );
      int ietOut = floor( et / params_->towerLsbE() );
      //      int ietOutMask = (int) pow(2,params_->towerNBitsE())-1;
      
      if (ietIn>0) 
	LogDebug("L1TDebug") << " ECAL TP : " << ieta << ", " << iphi << ", " << ietIn << ", " << et << ", " << ietOut << std::endl;

      int itow = CaloTools::caloTowerHash(ieta, iphi);
      towers.at(itow).setHwEtEm(ietOut);// & ietOutMask);

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
      towers.at(itow).setHwEtHad(ietOut);// & ietOutMask);
      //      towers.at(itow).setHwFGHad(ifg);

    }

    // now calculate remaining tower quantities
    for (int ieta=ietaMin_; ieta<ietaMax_+1; ieta++) {

      for (int iphi=iphiMin_; iphi<iphiMax_+1; iphi++) {

	if(!CaloTools::isValidIEtaIPhi(ieta,iphi)) continue;

	int itow = CaloTools::caloTowerHash(ieta, iphi);

	// get ECAL/HCAL raw numbers
	int ietEcal = towers.at(itow).hwEtEm();
	int ietHcal = towers.at(itow).hwEtHad();
	
	//	const LorentzVector& p4;
	int iet = ietEcal + ietHcal;   // this is nonsense, temp solution!

	//LogDebug("L1TDebug") << " Tower : " << ieta << ", " << iphi << ", " << iet << ", " << ietEcal << ", " << ietHcal << std::endl;

	towers.at(itow).setHwPt(iet);
	towers.at(itow).setHwEta(ieta);
	towers.at(itow).setHwPhi(iphi);

      }
    }

    // copy towers to BXVector
    // could do this more quickly with improved BXVector interface
    std::vector<l1t::CaloTower>::const_iterator towItr;
    for (towItr=towers.begin(); towItr!=towers.end(); ++towItr) {
      towersColl->push_back(bx, (*towItr) );
    }

    LogDebug("L1TDebug") << "BX=" << bx << ", N(ECAL)=" << nEcal << ", N(HCAL)=" << nHcal << ", N(Towers)=" << towersColl->size(bx) << std::endl;

  }

  iEvent.put(towersColl);
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
l1t::L1TCaloTowerProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
l1t::L1TCaloTowerProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------

void
l1t::L1TCaloTowerProducer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{

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

}

 
// ------------ method called when ending the processing of a run  ------------

void
l1t::L1TCaloTowerProducer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
l1t::L1TCaloTowerProducer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup cons
t&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
l1t::L1TCaloTowerProducer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&
)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
l1t::L1TCaloTowerProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::L1TCaloTowerProducer);
