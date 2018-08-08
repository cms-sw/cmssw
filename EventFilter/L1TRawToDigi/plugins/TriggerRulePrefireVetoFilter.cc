// -*- C++ -*-
//
// Package:    L1Trigger/PrefireAnalysis
// Class:      TriggerRulePrefireVetoFilter
// 
/**\class TriggerRulePrefireVetoFilter TriggerRulePrefireVetoFilter.cc L1Trigger/PrefireAnalysis/plugins/TriggerRulePrefireVetoFilter.cc

 Description: [one line class summary]

 Implementation:
    [Notes on implementation]
*/
//
// Original Author:  Nicholas Charles Smith
//         Created:  Mon, 28 May 2018 15:39:05 GMT
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/TCDS/interface/TCDSRecord.h"

//
// class declaration
//

class TriggerRulePrefireVetoFilter : public edm::stream::EDFilter<> {
  public:
    explicit TriggerRulePrefireVetoFilter(const edm::ParameterSet&);
    ~TriggerRulePrefireVetoFilter() override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void beginStream(edm::StreamID) override;
    bool filter(edm::Event&, const edm::EventSetup&) override;
    void endStream() override;

    //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
    //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
    //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

    // ----------member data ---------------------------
    edm::EDGetTokenT<TCDSRecord> tcdsRecordToken_;
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
TriggerRulePrefireVetoFilter::TriggerRulePrefireVetoFilter(const edm::ParameterSet& iConfig) :
  tcdsRecordToken_(consumes<TCDSRecord>(iConfig.getParameter<edm::InputTag>("tcdsRecordLabel")))
{
  //now do what ever initialization is needed

}


TriggerRulePrefireVetoFilter::~TriggerRulePrefireVetoFilter()
{
 
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
TriggerRulePrefireVetoFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<TCDSRecord> tcdsRecordH;
  iEvent.getByToken(tcdsRecordToken_, tcdsRecordH);
  const auto& tcdsRecord = *tcdsRecordH.product();

  uint64_t thisEvent = (tcdsRecord.getBXID()-1) + tcdsRecord.getOrbitNr()*3564ull;

  std::vector<uint64_t> eventHistory;
  for (auto&& l1a : tcdsRecord.getFullL1aHistory()) {
    eventHistory.push_back(thisEvent - ((l1a.getBXID()-1) + l1a.getOrbitNr()*3564ull));
  }

  // should be 16 according to TCDSRecord.h, we only care about the last 4
  if ( eventHistory.size() < 4 ) {
    edm::LogError("TriggerRulePrefireVetoFilter") << "Unexpectedly small L1A history from TCDSRecord";
  }

  // No more than 1 L1A in 3 BX
  if ( eventHistory[0] < 3ull ) {
    edm::LogError("TriggerRulePrefireVetoFilter") << "Found an L1A in an impossible location?! (1 in 3)";
  }
  if ( eventHistory[0] == 3ull ) return true;

  // No more than 2 L1As in 25 BX
  if ( eventHistory[0] < 25ull and eventHistory[1] < 25ull ) {
    edm::LogError("TriggerRulePrefireVetoFilter") << "Found an L1A in an impossible location?! (2 in 25)";
  }
  if ( eventHistory[0] < 25ull and eventHistory[1] == 25ull ) return true;

  // No more than 3 L1As in 100 BX
  if ( eventHistory[0] < 100ull and eventHistory[1] < 100ull and eventHistory[2] < 100ull ) {
    edm::LogError("TriggerRulePrefireVetoFilter") << "Found an L1A in an impossible location?! (3 in 100)";
  }
  if ( eventHistory[0] < 100ull and eventHistory[1] < 100ull and eventHistory[2] == 100ull ) return true;

  // No more than 4 L1As in 240 BX
  if ( eventHistory[0] < 240ull and eventHistory[1] < 240ull and eventHistory[2] < 240ull and eventHistory[3] < 240ull ) {
    edm::LogError("TriggerRulePrefireVetoFilter") << "Found an L1A in an impossible location?! (4 in 240)";
  }
  if ( eventHistory[0] < 240ull and eventHistory[1] < 240ull and eventHistory[2] < 240ull and eventHistory[3] == 240ull ) return true;

  return false;
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
TriggerRulePrefireVetoFilter::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
TriggerRulePrefireVetoFilter::endStream() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
TriggerRulePrefireVetoFilter::beginRun(edm::Run const&, edm::EventSetup const&)
{ 
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
TriggerRulePrefireVetoFilter::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
TriggerRulePrefireVetoFilter::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
TriggerRulePrefireVetoFilter::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TriggerRulePrefireVetoFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(TriggerRulePrefireVetoFilter);
