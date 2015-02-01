//-------------------------------------------------
//
/**  \class HLTL1MuonSelector
 * 
 *   HLTL1MuonSelector:
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
#include "HLTrigger/Muon/interface/HLTL1MuonSelector.h"


// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

using namespace std;
using namespace edm;
using namespace l1extra;

// constructors
HLTL1MuonSelector::HLTL1MuonSelector(const edm::ParameterSet& iConfig) : 
  theSource(iConfig.getParameter<InputTag>("InputObjects")),
  theL1MinPt(iConfig.getParameter<double>("L1MinPt")),
  theL1MaxEta(iConfig.getParameter<double>("L1MaxEta")),
  theL1MinQuality(iConfig.getParameter<unsigned int>("L1MinQuality"))
{
  muCollToken_ = consumes<L1MuonParticleCollection>(theSource);

  produces<L1MuonParticleCollection>(); 
}

// destructor
HLTL1MuonSelector::~HLTL1MuonSelector(){
}

void
HLTL1MuonSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputObjects",edm::InputTag(""));
  desc.add<double>("L1MinPt",-1.);
  desc.add<double>("L1MaxEta",5.0);
  desc.add<unsigned int>("L1MinQuality",0);
  descriptions.add("hltL1MuonSelector",desc);
}

void HLTL1MuonSelector::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
  const std::string metname = "Muon|RecoMuon|HLTL1MuonSelector";

  auto_ptr<L1MuonParticleCollection> output(new L1MuonParticleCollection());
  
  // Muon particles 
  edm::Handle<L1MuonParticleCollection> muColl;
  iEvent.getByToken(muCollToken_, muColl);
  LogTrace(metname) << "Number of muons " << muColl->size() << endl;

  L1MuonParticleCollection::const_iterator it;
  L1MuonParticleRef::key_type l1ParticleIndex = 0;

  for(it = muColl->begin(); it != muColl->end(); ++it,++l1ParticleIndex) {
    
    const L1MuGMTExtendedCand muonCand = (*it).gmtMuonCand();
    unsigned int quality = 0;
    bool valid_charge = false;;

    if ( muonCand.empty() ) {
      LogWarning(metname) << "HLTL1MuonSelector: WARNING, no L1MuGMTCand! " << endl;
      LogWarning(metname) << "HLTL1MuonSelector:   this should make sense only within MC tests" << endl;
      // FIXME! Temporary to handle the MC input
      quality = 7;
      valid_charge = true;
    }
    else {
      quality =  muonCand.quality();
      valid_charge = muonCand.charge_valid();
    }
    
    float pt    =  (*it).pt();
    float eta   =  (*it).eta();
    float theta =  2*atan(exp(-eta));
    float phi   =  (*it).phi();      
    int charge  =  (*it).charge();
    // Set charge=0 for the time being if the valid charge bit is zero
    if (!valid_charge) charge = 0;
    bool barrel = !(*it).isForward();

    if ( pt < theL1MinPt || fabs(eta) > theL1MaxEta ) continue;
    
    LogTrace(metname) << "L1 Muon Found";
    LogTrace(metname) << "Pt = " << pt << " GeV/c";
    LogTrace(metname) << "eta = " << eta;
    LogTrace(metname) << "theta = " << theta << " rad";
    LogTrace(metname) << "phi = " << phi << " rad";
    LogTrace(metname) << "charge = "<< charge;
    LogTrace(metname) << "In Barrel? = "<< barrel;
    
    if ( quality <= theL1MinQuality ) continue;
    LogTrace(metname) << "quality = "<< quality; 

    output->push_back( L1MuonParticle(*it) );
    
  } // loop over L1MuonParticleCollection
  
  iEvent.put(output);
}

