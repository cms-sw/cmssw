///
/// \class l1t::L1TCaloUpgrateToGCTConverter
///
/// Description: Emulator for the stage 1 jet algorithms.
///
///
/// \author: R. Alex Barbieri MIT
///


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

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage1DataFormatter.h"

using namespace std;
using namespace edm;

namespace l1t {

//
// class declaration
//

  class L1TCaloUpgrateToGCTConverter : public EDProducer {
  public:
    explicit L1TCaloUpgrateToGCTConverter(const ParameterSet&);
    ~L1TCaloUpgrateToGCTConverter();

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    virtual void produce(Event&, EventSetup const&);
    virtual void beginJob();
    virtual void endJob();
    virtual void beginRun(Run const&iR, EventSetup const&iE);
    virtual void endRun(Run const& iR, EventSetup const& iE);

      
      EDGetToken EGammaToken;
      EDGetToken TauToken;
      EDGetToken JetToken;
      EDGetToken EtSumToken;
      
      CaloStage1DataFormatter DataFormatter;
     
    };

  //
  // constructors and destructor
  //
  L1TCaloUpgrateToGCTConverter::L1TCaloUpgrateToGCTConverter(const ParameterSet& iConfig)
  {
   
   
    
    produces<L1GctEmCandCollection>("isoEm");
    produces<L1GctEmCandCollection>("nonIsoEm");
    produces<L1GctJetCandCollection>("cenJets");
    produces<L1GctJetCandCollection>("forJets");
    produces<L1GctJetCandCollection>("tauJets");
    produces<L1GctInternJetDataCollection>();
    produces<L1GctEtTotalCollection>();
    produces<L1GctEtHadCollection>();
    produces<L1GctEtMissCollection>();
    produces<L1GctHtMissCollection>();
    produces<L1GctInternEtSumCollection>();
    produces<L1GctInternHtMissCollection>();
    produces<L1GctHFBitCountsCollection>();
    produces<L1GctHFRingEtSumsCollection>();
      
      // register what you consume and keep token for later access:
      EGammaToken = consumes<L1TEGammaCollection>(iConfig.getParameter<InputTag>("CaloRegions"));
      TauToken = consumes<L1TTauCollection>(iConfig.getParameter<InputTag>("CaloEmCands"));
      JetToken = consumes<L1TJetCollection>(iConfig.getParameter<InputTag>("CaloRegions"));
      EtSumToken = consumes<L1TEtSumCollection>(iConfig.getParameter<InputTag>("CaloEmCands"));

   
  }


  L1TCaloUpgrateToGCTConverter::~L1TCaloUpgrateToGCTConverter()
  {
  }



//
// member functions
//

// ------------ method called to produce the data ------------
void
L1TCaloUpgrateToGCTConverter::produce(Event& e, const EventSetup& es)
{

  LogDebug("l1t|stage 1 Converter") << "L1TCaloUpgrateToGCTConverter::produce function called...\n";

  //inputs
    Handle<L1TEGammaCollection> EGamma;
    e.getByToken(EGammaToken,EGamma);

  Handle<L1TTauCollection> Tau;
  e.getByToken(TauToken,Tau);

  Handle<L1TJetCollection> Jet;
  e.getByToken(JetToken,Jet);
    
  Handle<L1TEtSumCollection> EtSum;
  e.getByToken(EtSumToken,EtSum);
  
    
    // create the em and jet collections
    std::auto_ptr<L1GctEmCandCollection> isoEmResult   (new L1GctEmCandCollection( ) );
    std::auto_ptr<L1GctEmCandCollection> nonIsoEmResult(new L1GctEmCandCollection( ) );
    std::auto_ptr<L1GctJetCandCollection> cenJetResult(new L1GctJetCandCollection( ) );
    std::auto_ptr<L1GctJetCandCollection> forJetResult(new L1GctJetCandCollection( ) );
    std::auto_ptr<L1GctJetCandCollection> tauJetResult(new L1GctJetCandCollection( ) );
    
    // create the energy sum digis
    std::auto_ptr<L1GctEtTotalCollection> etTotResult (new L1GctEtTotalCollection( ) );
    std::auto_ptr<L1GctEtHadCollection>   etHadResult (new L1GctEtHadCollection  ( ) );
    std::auto_ptr<L1GctEtMissCollection>  etMissResult(new L1GctEtMissCollection ( ) );
    std::auto_ptr<L1GctHtMissCollection>  htMissResult(new L1GctHtMissCollection ( ) );
    
    // create the Hf sums digis
    std::auto_ptr<L1GctHFBitCountsCollection>  hfBitCountResult (new L1GctHFBitCountsCollection ( ) );
    std::auto_ptr<L1GctHFRingEtSumsCollection> hfRingEtSumResult(new L1GctHFRingEtSumsCollection( ) );
    
    // create internal data collections
    std::auto_ptr<L1GctInternJetDataCollection> internalJetResult   (new L1GctInternJetDataCollection( ));
    std::auto_ptr<L1GctInternEtSumCollection>   internalEtSumResult (new L1GctInternEtSumCollection  ( ));
    std::auto_ptr<L1GctInternHtMissCollection>  internalHtMissResult(new L1GctInternHtMissCollection ( ));
    
    
    DataFormatter.ConvertToIsoEmCand(*EGamma, *isoEmResult);
   // DataFormatter.ConvertToNonIsoEmCand(EGamma, nonIsoEmResult);
   // DataFormatter.ConvertToCenJetCand(Jet, cenJetResult);
   // DataFormatter.ConvertToForJetCand(Jet, forJetResult);
   // DataFormatter.ConvertToTauJetCand(Tau, tauJetResult);

  //  DataFormatter.ConvertToEtTotal(EtSum, etTotResult);
  // DataFormatter.ConvertToEtHad(EtSum,etHadResult);
  // DataFormatter.ConvertToEtMiss(EtSum,etMissResult);
  // DataFormatter.ConvertToHtMiss(EtSum,htMissResult);
  // DataFormatter.ConvertToHFBitCounts(EtSum,hfBitCountResult);
  // DataFormatter.ConvertToHFRingEtSums(EtSum, hfRingEtSumResult);
    
  //  DataFormatter.ConvertToIntJet(Jet, internalJetResult);
  //  DataFormatter.ConvertToIntEtSum(EtSum,internalEtSumResult);
  //  DataFormatter.ConvertToIntHtMiss(EtSum,internalHtMissResult);

    
 

    
    e.put(isoEmResult,"isoEm");
    e.put(nonIsoEmResult,"nonIsoEm");
    e.put(cenJetResult,"cenJets");
    e.put(forJetResult,"forJets");
    e.put(tauJetResult,"tauJets");
    e.put(etTotResult);
    e.put(etHadResult);
    e.put(etMissResult);
    e.put(htMissResult);
    e.put(hfBitCountResult);
    e.put(hfRingEtSumResult);
    
    e.put(internalJetResult);
    e.put(internalEtSumResult);
    e.put(internalHtMissResult);
    
}

// ------------ method called once each job just before starting event loop ------------
void
L1TCaloUpgrateToGCTConverter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop ------------
void
L1TCaloUpgrateToGCTConverter::endJob() {
}

// ------------ method called when starting to processes a run ------------

void L1TCaloUpgrateToGCTConverter::beginRun(Run const&iR, EventSetup const&iE){

}

// ------------ method called when ending the processing of a run ------------
void L1TCaloUpgrateToGCTConverter::endRun(Run const& iR, EventSetup const& iE){

}


// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void
L1TCaloUpgrateToGCTConverter::fillDescriptions(ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

} // namespace

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::L1TCaloUpgrateToGCTConverter);
