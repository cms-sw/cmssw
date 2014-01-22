#ifndef L1TCaloUpgrateToGCTConverter_h
#define L1TCaloUpgrateToGCTConverter_h

///
/// \class l1t::L1TCaloUpgrateToGCTConverter
///
/// Description: Emulator for the stage 1 jet algorithms.
///
///
/// \author: Ivan Amos Cali MIT
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

#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

#include <vector>

typedef BXVector<l1t::EGamma> L1TEGammaCollection;
typedef BXVector<l1t::Tau> L1TTauCollection;
typedef BXVector<l1t::Jet> L1TJetCollection;
typedef BXVector<l1t::EtSum> L1TEtSumCollection;

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
    virtual void produce(Event&, EventSetup const&) override;
    virtual void beginJob();
    virtual void endJob();
    virtual void beginRun(Run const&iR, EventSetup const&iE);
    virtual void endRun(Run const& iR, EventSetup const& iE);

      L1GctEmCandCollection ConvertToIsoEmCand(L1TEGammaCollection::const_iterator);
    
      
    L1GctEmCandCollection ConvertToNonIsoEmCand(const L1TEGammaCollection&);
    L1GctJetCandCollection ConvertToCenJetCand(const L1TJetCollection&);
    L1GctJetCandCollection ConvertToForJetCand(const L1TJetCollection&);
    L1GctJetCandCollection ConvertToTauJetCand(const L1TTauCollection&);
      
    L1GctEtTotalCollection ConvertToEtTotal(const L1TEtSumCollection&);
    L1GctEtHadCollection ConvertToEtHad(const L1TEtSumCollection&);
    L1GctEtMissCollection ConvertToEtMiss(const L1TEtSumCollection&);
    L1GctHtMissCollection ConvertToHtMiss(const L1TEtSumCollection&);
    L1GctHFBitCountsCollection ConvertToHFBitCounts(const L1TEtSumCollection&);
    L1GctHFRingEtSumsCollection ConvertToHFRingEtSums(const L1TEtSumCollection&);
      
    L1GctInternJetDataCollection ConvertToIntJet(const L1TJetCollection&);
    L1GctInternEtSumCollection ConvertToIntEtSum(const L1TEtSumCollection&);
    L1GctInternHtMissCollection ConvertToIntHtMiss(const L1TEtSumCollection&);
      
    EDGetToken EGammaToken_;
    EDGetToken TauToken_;
    EDGetToken JetToken_;
    EDGetToken EtSumToken_;
      
     
};
}
#endif
 
//define this as a plug-in
DEFINE_FWK_MODULE(l1t::L1TCaloUpgrateToGCTConverter);
