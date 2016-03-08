//-------------------------------------------------
//
/**  \class HLTL1TMuonSelector
 * 
 *   HLTL1TMuonSelector:
 *   Simple selector to output a subset of L1 muon collection 
 *   
 *   based on RecoMuon/L2MuonSeedGenerator
 *
 *
 *   \author  D. Olivito
 */
//
//--------------------------------------------------

// Class Header
#include "HLTrigger/Muon/interface/HLTL1TMuonSelector.h"


// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

using namespace std;
using namespace edm;
using namespace l1t;

// constructors
HLTL1TMuonSelector::HLTL1TMuonSelector(const edm::ParameterSet& iConfig) : 
  theSource(iConfig.getParameter<InputTag>("InputObjects")),
  theL1MinPt(iConfig.getParameter<double>("L1MinPt")),
  theL1MaxEta(iConfig.getParameter<double>("L1MaxEta")),
  theL1MinQuality(iConfig.getParameter<unsigned int>("L1MinQuality")),
  centralBxOnly_( iConfig.getParameter<bool>("CentralBxOnly") )
{
  muCollToken_ = consumes<MuonBxCollection>(theSource);

  produces<MuonBxCollection>(); 
}

// destructor
HLTL1TMuonSelector::~HLTL1TMuonSelector(){
}

void
HLTL1TMuonSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputObjects",edm::InputTag("hltGmtStage2Digis"));
  desc.add<double>("L1MinPt",-1.);
  desc.add<double>("L1MaxEta",5.0);
  desc.add<unsigned int>("L1MinQuality",0);
  desc.add<bool>("CentralBxOnly", true);
  descriptions.add("hltL1TMuonSelector",desc);
}

void HLTL1TMuonSelector::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
  const std::string metname = "Muon|RecoMuon|HLTL1TMuonSelector";

  auto_ptr<MuonBxCollection> output(new MuonBxCollection());
  
  // Muon particles 
  edm::Handle<MuonBxCollection> muColl;
  iEvent.getByToken(muCollToken_, muColl);
  LogTrace(metname) << "Number of muons " << muColl->size() << endl;

  for (int ibx = muColl->getFirstBX(); ibx <= muColl->getLastBX(); ++ibx) {
    if (centralBxOnly_ && (ibx != 0)) continue;
    for (auto it = muColl->begin(ibx); it != muColl->end(ibx); it++){
    
      unsigned int quality = it->hwQual();
      int valid_charge = it->hwChargeValid();
    
      float pt    =  it->pt();
      float eta   =  it->eta();
      float theta =  2*atan(exp(-eta));
      float phi   =  it->phi();      
      int charge  =  it->charge();
      // Set charge=0 for the time being if the valid charge bit is zero
      if (!valid_charge) charge = 0;
    
      if ( pt < theL1MinPt || fabs(eta) > theL1MaxEta ) continue;
  
      LogTrace(metname) << "L1 Muon Found";
      LogTrace(metname) << "Pt = "     << pt    << " GeV/c";
      LogTrace(metname) << "eta = "    << eta;
      LogTrace(metname) << "theta = "  << theta << " rad";
      LogTrace(metname) << "phi = "    << phi   << " rad";
      LogTrace(metname) << "charge = " << charge;

      if ( quality <= theL1MinQuality ) continue;
      LogTrace(metname) << "quality = "<< quality; 

      output->push_back( ibx, *it);
    }
  }


  iEvent.put(output);
}

