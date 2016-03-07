// -*- C++ -*-
//
// Package:    L1Analysis/SelectL1Muons
// Class:      SelectL1Muons
//
/**\class SelectL1Muons SelectL1Muons.cc L1Analysis/SelectL1Muons/plugins/SelectL1Muons.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Joschka Philip Lingemann
//         Created:  Mon, 13 Jul 2015 18:45:16 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
//
// class declaration
//

class SelectL1Muons : public edm::EDFilter {
   public:
      explicit SelectL1Muons(const edm::ParameterSet&);
      ~SelectL1Muons();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() override;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
      edm::InputTag ugmtTag;
      edm::InputTag gmtTag;

      edm::EDGetTokenT<l1t::MuonBxCollection> ugmtToken;
      edm::EDGetTokenT<L1MuGMTReadoutCollection> gmtToken;
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
SelectL1Muons::SelectL1Muons(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  ugmtTag = iConfig.getParameter<edm::InputTag>("ugmtInput");
  ugmtToken = consumes<l1t::MuonBxCollection>(ugmtTag);

  gmtTag = iConfig.getParameter<edm::InputTag>("gmtInput");
  gmtToken = consumes<L1MuGMTReadoutCollection>(gmtTag);
}


SelectL1Muons::~SelectL1Muons()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
SelectL1Muons::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   Handle<l1t::MuonBxCollection> pMuonColl;
   iEvent.getByToken(ugmtToken, pMuonColl);

   bool haveUGMT = false;

   for (int bx = pMuonColl->getFirstBX(); bx <= pMuonColl->getLastBX(); ++bx) {
      if (pMuonColl->size(bx) > 0) {
         haveUGMT = true;
         break;
      }
   }

   Handle<L1MuGMTReadoutCollection> pGMTColl;
   iEvent.getByToken(gmtToken, pGMTColl);
   // L1MuGMTReadoutCollection const* gmtrc = gmtrc_handle.product();
   std::vector<L1MuGMTReadoutRecord> gmt_records = pGMTColl->getRecords();

   bool haveGMT = false;
   std::vector<L1MuGMTReadoutRecord>::const_iterator igmtrr;
   for(igmtrr=gmt_records.begin(); igmtrr!=gmt_records.end(); igmtrr++) {
      std::vector<L1MuGMTExtendedCand> exc = igmtrr->getGMTCands();

      if (exc.size() > 0) {
       haveGMT = true;
       break;
      }
   }

   return (haveGMT || haveUGMT);
}

// ------------ method called once each job just before starting event loop  ------------
void
SelectL1Muons::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
SelectL1Muons::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
SelectL1Muons::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
SelectL1Muons::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
SelectL1Muons::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
SelectL1Muons::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
SelectL1Muons::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(SelectL1Muons);
