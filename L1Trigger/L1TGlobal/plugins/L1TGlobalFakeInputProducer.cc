///
/// \class l1t::L1TGlobalFakeInputProducer
///
/// Description: Create Fake Input Collections for the GT.  Allows testing of emulation
///
/// 
/// \author: B. Winer OSU
///
///  Modeled after L1TGlobalFakeInputProducer


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

//#include <vector>
#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

using namespace std;
using namespace edm;

namespace l1t {

//
// class declaration
//

  class L1TGlobalFakeInputProducer : public EDProducer {
  public:
    explicit L1TGlobalFakeInputProducer(const ParameterSet&);
    ~L1TGlobalFakeInputProducer();

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    virtual void produce(Event&, EventSetup const&);
    virtual void beginJob();
    virtual void endJob();
    virtual void beginRun(Run const&iR, EventSetup const&iE);
    virtual void endRun(Run const& iR, EventSetup const& iE);

    // ----------member data ---------------------------
    unsigned long long m_paramsCacheId; // Cache-ID from current parameters, to check if needs to be updated.
    //boost::shared_ptr<const CaloParams> m_dbpars; // Database parameters for the trigger, to be updated as needed.
    //boost::shared_ptr<const FirmwareVersion> m_fwv;
    //boost::shared_ptr<FirmwareVersion> m_fwv; //not const during testing.

  };

  //
  // constructors and destructor
  //
  L1TGlobalFakeInputProducer::L1TGlobalFakeInputProducer(const ParameterSet& iConfig)
  {
    // register what you produce
    produces<BXVector<l1t::EGamma>>();
    produces<BXVector<l1t::Muon>>();
    produces<BXVector<l1t::Tau>>();
    produces<BXVector<l1t::Jet>>();
    produces<BXVector<l1t::EtSum>>();

    // set cache id to zero, will be set at first beginRun:
    m_paramsCacheId = 0;
  }


  L1TGlobalFakeInputProducer::~L1TGlobalFakeInputProducer()
  {
  }



//
// member functions
//

// ------------ method called to produce the data ------------
void
L1TGlobalFakeInputProducer::produce(Event& iEvent, const EventSetup& iSetup)
{

  LogDebug("l1t|Global") << "L1TGlobalFakeInputProducer::produce function called...\n";

  // Set the range of BX
  int bxFirst = -2;
  int bxLast  = 2;

  

  //outputs
  std::auto_ptr<l1t::EGammaBxCollection> egammas (new l1t::EGammaBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<l1t::MuonBxCollection> muons (new l1t::MuonBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<l1t::TauBxCollection> taus (new l1t::TauBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<l1t::JetBxCollection> jets (new l1t::JetBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<l1t::EtSumBxCollection> etsums (new l1t::EtSumBxCollection(0, bxFirst, bxLast));

   printf("I am in the producer...\n");
  
   //Loop over the bx
   for(int i = bxFirst; i <= bxLast; i++) {
     std::auto_ptr< std::vector<l1t::EGamma> > localEGammas (new std::vector<l1t::EGamma>);
     std::auto_ptr< std::vector<l1t::Muon> > localMuons (new std::vector<l1t::Muon>);
     std::auto_ptr< std::vector<l1t::Tau> > localTaus (new std::vector<l1t::Tau>);
     std::auto_ptr< std::vector<l1t::Jet> > localJets (new std::vector<l1t::Jet>);
     std::auto_ptr< std::vector<l1t::EtSum> > localEtSums (new std::vector<l1t::EtSum>);

// Simple hand-coded fake data for starters.
     int egPt   = 40 + i*10; //30 - 40 - 50 for bx -1 0 1
     int egEta  = 20;
     int egPhi  = 19;
     int egQual = 1;
     int egIso  = 0;
     ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *egLorentz = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
     l1t::EGamma fakeEG(*egLorentz, egPt, egEta, egPhi, egQual, egIso); 
     localEGammas->push_back(fakeEG);       
    
     for(std::vector<l1t::EGamma>::const_iterator eg = localEGammas->begin(); eg != localEGammas->end(); ++eg)  egammas->push_back(i, *eg);
     for(std::vector<l1t::Tau>::const_iterator tau = localTaus->begin(); tau != localTaus->end(); ++tau) taus->push_back(i, *tau);
     for(std::vector<l1t::Jet>::const_iterator jet = localJets->begin(); jet != localJets->end(); ++jet) jets->push_back(i, *jet);
     for(std::vector<l1t::EtSum>::const_iterator etsum = localEtSums->begin(); etsum != localEtSums->end(); ++etsum) etsums->push_back(i, *etsum);   
   
   }


  iEvent.put(egammas);
  iEvent.put(taus);
  iEvent.put(jets);
  iEvent.put(etsums);

}

// ------------ method called once each job just before starting event loop ------------
void
L1TGlobalFakeInputProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop ------------
void
L1TGlobalFakeInputProducer::endJob() {
}

// ------------ method called when starting to processes a run ------------

void L1TGlobalFakeInputProducer::beginRun(Run const&iR, EventSetup const&iE){

  LogDebug("l1t|Global") << "L1TGlobalFakeInputProducer::beginRun function called...\n";


}

// ------------ method called when ending the processing of a run ------------
void L1TGlobalFakeInputProducer::endRun(Run const& iR, EventSetup const& iE){

}


// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void
L1TGlobalFakeInputProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

} // namespace

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::L1TGlobalFakeInputProducer);
