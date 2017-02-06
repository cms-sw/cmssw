// -*- C++ -*-
//
// L1TMuonQualityAdjuster
//
// Fictitious module which filters/remaps TF qualities for studies
//
//

// system include files
#include <memory>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

using namespace l1t;

#include <iostream>
//
// class declaration
//

class L1TMuonQualityAdjuster : public edm::EDProducer {
public:
  explicit L1TMuonQualityAdjuster(const edm::ParameterSet&);
  ~L1TMuonQualityAdjuster();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() override ;
  virtual void produce(edm::Event&, const edm::EventSetup&) override ;
  virtual void endJob() override ;
  virtual void beginRun(const edm::Run&, edm::EventSetup const&) override ;
  virtual void endRun(const edm::Run&, edm::EventSetup const&) override ;
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, edm::EventSetup const&) override ;
  virtual void endLuminosityBlock(const edm::LuminosityBlock&, edm::EventSetup const&) override ;
  // ----------member data ---------------------------
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> m_barrelTfInputToken;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> m_overlapTfInputToken;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> m_endCapTfInputToken;
  // edm::InputTag m_barrelTfInputTag;
  edm::InputTag m_barrelTfInputTag;
  edm::InputTag m_overlapTfInputTag;
  edm::InputTag m_endCapTfInputTag;
  int m_bmtfBxOffset; // Hack while sorting our source of -3 BX offset when re-Emulating...
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
L1TMuonQualityAdjuster::L1TMuonQualityAdjuster(const edm::ParameterSet& iConfig)
{
  m_barrelTfInputTag  = iConfig.getParameter<edm::InputTag>("bmtfInput");
  m_overlapTfInputTag = iConfig.getParameter<edm::InputTag>("omtfInput");
  m_endCapTfInputTag  = iConfig.getParameter<edm::InputTag>("emtfInput");
  m_bmtfBxOffset      = iConfig.getParameter<int>("bmtfBxOffset");
  m_barrelTfInputToken  = consumes<l1t::RegionalMuonCandBxCollection>(m_barrelTfInputTag);
  m_overlapTfInputToken = consumes<l1t::RegionalMuonCandBxCollection>(m_overlapTfInputTag);
  m_endCapTfInputToken  = consumes<l1t::RegionalMuonCandBxCollection>(m_endCapTfInputTag);
  //register your products
  produces<RegionalMuonCandBxCollection>("BMTF");
  produces<RegionalMuonCandBxCollection>("OMTF");
  produces<RegionalMuonCandBxCollection>("EMTF");
}


L1TMuonQualityAdjuster::~L1TMuonQualityAdjuster()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//


// ------------ method called to produce the data  ------------
void
L1TMuonQualityAdjuster::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  std::auto_ptr<l1t::RegionalMuonCandBxCollection> filteredBMTFMuons (new l1t::RegionalMuonCandBxCollection());
  std::auto_ptr<l1t::RegionalMuonCandBxCollection> filteredOMTFMuons (new l1t::RegionalMuonCandBxCollection());
  std::auto_ptr<l1t::RegionalMuonCandBxCollection> filteredEMTFMuons (new l1t::RegionalMuonCandBxCollection());

  Handle<l1t::RegionalMuonCandBxCollection> bmtfMuons;
  Handle<l1t::RegionalMuonCandBxCollection> omtfMuons;
  Handle<l1t::RegionalMuonCandBxCollection> emtfMuons;

  iEvent.getByToken(m_barrelTfInputToken, bmtfMuons);
  iEvent.getByToken(m_overlapTfInputToken, omtfMuons);
  iEvent.getByToken(m_endCapTfInputToken, emtfMuons);

  
  if (bmtfMuons.isValid()){
    filteredBMTFMuons->setBXRange(bmtfMuons->getFirstBX(), bmtfMuons->getLastBX());
    for (int bx = bmtfMuons->getFirstBX(); bx <= bmtfMuons->getLastBX(); ++bx) {
      for (auto mu = bmtfMuons->begin(bx); mu != bmtfMuons->end(bx); ++mu) {
	int newqual = 12;
	l1t::RegionalMuonCand newMu((*mu));      
	newMu.setHwQual(newqual);
	filteredBMTFMuons->push_back(bx+m_bmtfBxOffset, newMu);      
      }
    }
  } else {
    //cout << "ERROR:  did not find BMTF muons in event with label:  " << m_barrelTfInputTag << "\n";
  }

  if (emtfMuons.isValid()){
    filteredEMTFMuons->setBXRange(emtfMuons->getFirstBX(), emtfMuons->getLastBX());
    for (int bx = emtfMuons->getFirstBX(); bx <= emtfMuons->getLastBX(); ++bx) {
      for (auto mu = emtfMuons->begin(bx); mu != emtfMuons->end(bx); ++mu) {
	int newqual = 0;
	if (mu->hwQual() == 11 || mu->hwQual() > 12) newqual=12;
	l1t::RegionalMuonCand newMu((*mu));
	newMu.setHwQual(newqual);
	filteredEMTFMuons->push_back(bx, newMu);
      }    
    }
  } else {
    //cout << "ERROR:  did not find EMTF muons in event with label:  " << m_endCapTfInputTag << "\n";
  }

  if (omtfMuons.isValid()){
    filteredOMTFMuons->setBXRange(omtfMuons->getFirstBX(), omtfMuons->getLastBX());
    for (int bx = omtfMuons->getFirstBX(); bx <= omtfMuons->getLastBX(); ++bx) {
      for (auto mu = omtfMuons->begin(bx); mu != omtfMuons->end(bx); ++mu) {
	int newqual = 0;
	if (mu->hwQual() > 0) newqual = 12;
	l1t::RegionalMuonCand newMu((*mu));
	newMu.setHwQual(newqual);
	filteredOMTFMuons->push_back(bx, newMu);
      }
    } 
  } else {
    //cout << "ERROR:  did not find OMTF muons in event with label:  " << m_overlapTfInputTag << "\n";
  }

  //cout << "Size BMTF(-3):  " << filteredBMTFMuons->size(-3) << "\n";
  //cout << "Size BMTF:  " << filteredBMTFMuons->size(0) << "\n";
  //cout << "Size OMTF:  " << filteredOMTFMuons->size(0) << "\n";
  //cout << "Size EMTF:  " << filteredEMTFMuons->size(0) << "\n";

  iEvent.put(filteredBMTFMuons, "BMTF");
  iEvent.put(filteredOMTFMuons, "OMTF");
  iEvent.put(filteredEMTFMuons, "EMTF");
}

// ------------ method called once each job just before starting event loop  ------------
void
L1TMuonQualityAdjuster::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TMuonQualityAdjuster::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
L1TMuonQualityAdjuster::beginRun(const edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void
L1TMuonQualityAdjuster::endRun(const edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void
L1TMuonQualityAdjuster::beginLuminosityBlock(const edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void
L1TMuonQualityAdjuster::endLuminosityBlock(const edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TMuonQualityAdjuster::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}



//define this as a plug-in
DEFINE_FWK_MODULE(L1TMuonQualityAdjuster);
