// L1TCaloUpgradeToGCTConverter.cc
// Authors: Ivan Cali
//          R. Alex Barbieri

// Stage 1 upgrade to old GT format converter
// Assumes input collections are sorted, but not truncated.

// In the 'gct' eta coordinates the HF is 0-3 and 18-21. Jets which
// include any energy at all from the HF should be considered
// 'forward' jets, however, so jets with centers in 0-4 and 17-21 are
// considered 'forward'.


#include "L1Trigger/L1TCalorimeter/plugins/L1TCaloUpgradeToGCTConverter.h"
#include <boost/shared_ptr.hpp>

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

l1t::L1TCaloUpgradeToGCTConverter::L1TCaloUpgradeToGCTConverter(const ParameterSet& iConfig)
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
  EGammaToken_ = consumes<l1t::EGammaBxCollection>(iConfig.getParameter<InputTag>("InputCollection"));
  TauToken_ = consumes<l1t::TauBxCollection>(iConfig.getParameter<InputTag>("InputCollection"));
  JetToken_ = consumes<l1t::JetBxCollection>(iConfig.getParameter<InputTag>("InputCollection"));
  EtSumToken_ = consumes<l1t::EtSumBxCollection>(iConfig.getParameter<InputTag>("InputCollection"));
}


l1t::L1TCaloUpgradeToGCTConverter::~L1TCaloUpgradeToGCTConverter()
{
}




// ------------ method called to produce the data ------------
void
l1t::L1TCaloUpgradeToGCTConverter::produce(Event& e, const EventSetup& es)
{
  LogDebug("l1t|stage 1 Converter") << "L1TCaloUpgradeToGCTConverter::produce function called...\n";

  //inputs
  Handle<l1t::EGammaBxCollection> EGamma;
  e.getByToken(EGammaToken_,EGamma);

  Handle<l1t::TauBxCollection> Tau;
  e.getByToken(TauToken_,Tau);

  Handle<l1t::JetBxCollection> Jet;
  e.getByToken(JetToken_,Jet);

  Handle<l1t::EtSumBxCollection> EtSum;
  e.getByToken(EtSumToken_,EtSum);

  // create the em and jet collections
  std::auto_ptr<L1GctEmCandCollection> isoEmResult(new L1GctEmCandCollection( ) );
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


  // Assume BX is the same for all collections
  int firstBX = EGamma->getFirstBX();
  int lastBX = EGamma->getLastBX();
  // //Finding the BX range for each input collection
  // int firstBxEGamma = EGamma->getFirstBX();
  // int lastBxEGamma = EGamma->getLastBX();
  // int firstBxTau = Tau->getFirstBX();
  // int lastBxTau = Tau->getLastBX();
  // int firstBxJet = Jet->getFirstBX();
  // int lastBxJet = Jet->getLastBX();
  // int firstBxEtSum = EtSum->getFirstBX();
  // int lastBxEtSum = EtSum->getLastBX();

  for(int itBX=firstBX; itBX!=lastBX+1; ++itBX){

    //looping over EGamma elments with a specific BX
    int nonIsoCount = 0;
    int isoCount = 0;
    for(l1t::EGammaBxCollection::const_iterator itEGamma = EGamma->begin(itBX);
	itEGamma != EGamma->end(itBX); ++itEGamma){
      bool iso = itEGamma->hwIso();

      L1GctEmCand EmCand(itEGamma->hwPt(), itEGamma->hwPhi(), itEGamma->hwEta(),
			 iso, 0, 0, itBX);
      //L1GctEmCand(unsigned rank, unsigned phi, unsigned eta,
      //                 bool iso, uint16_t block, uint16_t index, int16_t bx);

      if(iso){
	if(isoCount != 4)
	{
	  isoEmResult->push_back(EmCand);
	  isoCount++;
	}
      }
      else{
	if(nonIsoCount != 4)
	{
	  nonIsoEmResult->push_back(EmCand);
	  nonIsoCount++;
	}
      }
    }

    //looping over Tau elments with a specific BX
    int tauCount = 0; //max 4
    for(l1t::TauBxCollection::const_iterator itTau = Tau->begin(itBX);
	itTau != Tau->end(itBX); ++itTau){
      // taus are not allowed to be forward
      const bool forward= false;

      L1GctJetCand TauCand(itTau->hwPt(), itTau->hwPhi(), itTau->hwEta(),
			   true, forward,0, 0, itBX);
      //L1GctJetCand(unsigned rank, unsigned phi, unsigned eta,
      //             bool isTau, bool isFor, uint16_t block, uint16_t index, int16_t bx);
      if(tauCount != 4){
	tauJetResult->push_back(TauCand);
	tauCount++;
      }
    }

    //looping over Jet elments with a specific BX
    int forCount = 0; //max 4
    int cenCount = 0; //max 4
    for(l1t::JetBxCollection::const_iterator itJet = Jet->begin(itBX);
	itJet != Jet->end(itBX); ++itJet){
      // use 2nd quality bit to define forward
      const bool forward = ((itJet->hwQual() & 0x2) != 0);
      L1GctJetCand JetCand(itJet->hwPt(), itJet->hwPhi(), itJet->hwEta(),
			   false, forward,0, 0, itBX);
      //L1GctJetCand(unsigned rank, unsigned phi, unsigned eta,
      //             bool isTau, bool isFor, uint16_t block, uint16_t index, int16_t bx);
      if(forward) {
	if(forCount !=4 ){
	  forJetResult->push_back(JetCand);
	  forCount++;
	}
      }
      else {
	if(cenCount != 4){
	  cenJetResult->push_back(JetCand);
	  cenCount++;
	}
      }
    }

    //looping over EtSum elments with a specific BX
    for (l1t::EtSumBxCollection::const_iterator itEtSum = EtSum->begin(itBX);
	itEtSum != EtSum->end(itBX); ++itEtSum){

      if (EtSum::EtSumType::kMissingEt == itEtSum->getType()){
	L1GctEtMiss Cand(itEtSum->hwPt(), itEtSum->hwPhi(), 0, itBX);
	etMissResult->push_back(Cand);
      }else if (EtSum::EtSumType::kMissingHt == itEtSum->getType()){
	L1GctHtMiss Cand(itEtSum->hwPt(), itEtSum->hwPhi(), 0, itBX);
	htMissResult->push_back(Cand);
      }else if (EtSum::EtSumType::kTotalEt == itEtSum->getType()){
	L1GctEtTotal Cand(itEtSum->hwPt(), 0, itBX);
	etTotResult->push_back(Cand);
      }else if (EtSum::EtSumType::kTotalHt == itEtSum->getType()){
	L1GctEtHad Cand(itEtSum->hwPt(), 0, itBX);
	etHadResult->push_back(Cand);
      }else {
	LogError("l1t|stage 1 Converter") <<" Unknown EtSumType --- EtSum collection will not be saved...\n ";
      }
    }
  }

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
l1t::L1TCaloUpgradeToGCTConverter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop ------------
void
l1t::L1TCaloUpgradeToGCTConverter::endJob() {
}

// ------------ method called when starting to processes a run ------------

void
l1t::L1TCaloUpgradeToGCTConverter::beginRun(Run const&iR, EventSetup const&iE){

}

// ------------ method called when ending the processing of a run ------------
void
l1t::L1TCaloUpgradeToGCTConverter::endRun(Run const& iR, EventSetup const& iE){

}


// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void
l1t::L1TCaloUpgradeToGCTConverter::fillDescriptions(ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::L1TCaloUpgradeToGCTConverter);
