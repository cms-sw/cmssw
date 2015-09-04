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

using namespace std;
using namespace edm;
using namespace l1t;

L1TCaloUpgradeToGCTConverter::L1TCaloUpgradeToGCTConverter(const ParameterSet& iConfig):
    // register what you consume and keep token for later access:
    EGammaToken_(   consumes<EGammaBxCollection>(iConfig.getParameter<InputTag>("InputCollection")) ),
    RlxTauToken_(   consumes<TauBxCollection>(iConfig.getParameter<InputTag>("InputRlxTauCollection")) ),
    IsoTauToken_(   consumes<TauBxCollection>(iConfig.getParameter<InputTag>("InputIsoTauCollection")) ),
    JetToken_(      consumes<JetBxCollection>(iConfig.getParameter<InputTag>("InputCollection")) ),
    EtSumToken_(    consumes<EtSumBxCollection>(iConfig.getParameter<InputTag>("InputCollection")) ),
    HfSumsToken_(   consumes<CaloSpareBxCollection>(iConfig.getParameter<edm::InputTag>("InputHFSumsCollection")) ),
    HfCountsToken_( consumes<CaloSpareBxCollection>(iConfig.getParameter<edm::InputTag>("InputHFCountsCollection")) ),
    bxMin_(         iConfig.getParameter<int>("bxMin") ),
    bxMax_(         iConfig.getParameter<int>("bxMax") )
{
  produces<L1GctEmCandCollection>("isoEm");
  produces<L1GctEmCandCollection>("nonIsoEm");
  produces<L1GctJetCandCollection>("cenJets");
  produces<L1GctJetCandCollection>("forJets");
  produces<L1GctJetCandCollection>("tauJets");
  produces<L1GctJetCandCollection>("isoTauJets");
  produces<L1GctInternJetDataCollection>();
  produces<L1GctEtTotalCollection>();
  produces<L1GctEtHadCollection>();
  produces<L1GctEtMissCollection>();
  produces<L1GctHtMissCollection>();
  produces<L1GctInternEtSumCollection>();
  produces<L1GctInternHtMissCollection>();
  produces<L1GctHFBitCountsCollection>();
  produces<L1GctHFRingEtSumsCollection>();
}


// ------------ method called to produce the data ------------
void
L1TCaloUpgradeToGCTConverter::produce(StreamID, Event& e, const EventSetup& es) const
{
  LogDebug("l1t|stage 1 Converter") << "L1TCaloUpgradeToGCTConverter::produce function called...\n";

  //inputs
  Handle<EGammaBxCollection> EGamma;
  e.getByToken(EGammaToken_,EGamma);

  Handle<TauBxCollection> RlxTau;
  e.getByToken(RlxTauToken_,RlxTau);

  Handle<TauBxCollection> IsoTau;
  e.getByToken(IsoTauToken_,IsoTau);

  Handle<JetBxCollection> Jet;
  e.getByToken(JetToken_,Jet);

  Handle<EtSumBxCollection> EtSum;
  e.getByToken(EtSumToken_,EtSum);

  Handle<CaloSpareBxCollection> HfSums;
  e.getByToken(HfSumsToken_, HfSums);

  Handle<CaloSpareBxCollection> HfCounts;
  e.getByToken(HfCountsToken_, HfCounts);

  // create the em and jet collections
  std::auto_ptr<L1GctEmCandCollection> isoEmResult(new L1GctEmCandCollection( ) );
  std::auto_ptr<L1GctEmCandCollection> nonIsoEmResult(new L1GctEmCandCollection( ) );
  std::auto_ptr<L1GctJetCandCollection> cenJetResult(new L1GctJetCandCollection( ) );
  std::auto_ptr<L1GctJetCandCollection> forJetResult(new L1GctJetCandCollection( ) );
  std::auto_ptr<L1GctJetCandCollection> tauJetResult(new L1GctJetCandCollection( ) );
  std::auto_ptr<L1GctJetCandCollection> isoTauJetResult(new L1GctJetCandCollection( ) );

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

  int bxCounter = 0;

  for(int itBX=EGamma->getFirstBX(); itBX<=EGamma->getLastBX(); ++itBX){

    if (itBX<bxMin_) continue;
    if (itBX>bxMax_) continue;

    bxCounter++;

    //looping over EGamma elments with a specific BX
    int nonIsoCount = 0;
    int isoCount = 0;
    for(EGammaBxCollection::const_iterator itEGamma = EGamma->begin(itBX);
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
    isoEmResult->resize(4*bxCounter);
    nonIsoEmResult->resize(4*bxCounter);
  }

  bxCounter = 0;
  for(int itBX=RlxTau->getFirstBX(); itBX<=RlxTau->getLastBX(); ++itBX){

    if (itBX<bxMin_) continue;
    if (itBX>bxMax_) continue;

    bxCounter++;
    //looping over Tau elments with a specific BX
    int tauCount = 0; //max 4
    for(TauBxCollection::const_iterator itTau = RlxTau->begin(itBX);
	itTau != RlxTau->end(itBX); ++itTau){
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
    tauJetResult->resize(4*bxCounter);
  }

  bxCounter = 0;
  for(int itBX=IsoTau->getFirstBX(); itBX<=IsoTau->getLastBX(); ++itBX){

    if (itBX<bxMin_) continue;
    if (itBX>bxMax_) continue;

    bxCounter++;
    //looping over Iso Tau elments with a specific BX
    int isoTauCount = 0; //max 4
    for(TauBxCollection::const_iterator itTau = IsoTau->begin(itBX);
	itTau != IsoTau->end(itBX); ++itTau){
      // taus are not allowed to be forward
      const bool forward= false;

      L1GctJetCand TauCand(itTau->hwPt(), itTau->hwPhi(), itTau->hwEta(),
			   true, forward,0, 0, itBX);
      //L1GctJetCand(unsigned rank, unsigned phi, unsigned eta,
      //             bool isTau, bool isFor, uint16_t block, uint16_t index, int16_t bx);
      if(isoTauCount != 4){
	isoTauJetResult->push_back(TauCand);
	isoTauCount++;
      }
    }
    isoTauJetResult->resize(4*bxCounter);
  }

  bxCounter = 0;
  for(int itBX=Jet->getFirstBX(); itBX<=Jet->getLastBX(); ++itBX){

    if (itBX<bxMin_) continue;
    if (itBX>bxMax_) continue;

    bxCounter++;
    //looping over Jet elments with a specific BX
    int forCount = 0; //max 4
    int cenCount = 0; //max 4
    for(JetBxCollection::const_iterator itJet = Jet->begin(itBX);
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
    forJetResult->resize(4*bxCounter);
    cenJetResult->resize(4*bxCounter);
  }

  bxCounter = 0;
  for(int itBX=EtSum->getFirstBX(); itBX<=EtSum->getLastBX(); ++itBX){

    if (itBX<bxMin_) continue;
    if (itBX>bxMax_) continue;

    bxCounter++;
    //looping over EtSum elments with a specific BX
    for (EtSumBxCollection::const_iterator itEtSum = EtSum->begin(itBX);
	itEtSum != EtSum->end(itBX); ++itEtSum){

      if (EtSum::EtSumType::kMissingEt == itEtSum->getType()){
	L1GctEtMiss Cand(itEtSum->hwPt(), itEtSum->hwPhi(), itEtSum->hwQual()&0x1, itBX);
	etMissResult->push_back(Cand);
      }else if (EtSum::EtSumType::kMissingHt == itEtSum->getType()){
	L1GctHtMiss Cand(itEtSum->hwPt(), itEtSum->hwPhi(), itEtSum->hwQual()&0x1, itBX);
	htMissResult->push_back(Cand);
      }else if (EtSum::EtSumType::kTotalEt == itEtSum->getType()){
	L1GctEtTotal Cand(itEtSum->hwPt(), itEtSum->hwQual()&0x1, itBX);
	etTotResult->push_back(Cand);
      }else if (EtSum::EtSumType::kTotalHt == itEtSum->getType()){
	L1GctEtHad Cand(itEtSum->hwPt(), itEtSum->hwQual()&0x1, itBX);
	etHadResult->push_back(Cand);
      }else {
	LogError("l1t|stage 1 Converter") <<" Unknown EtSumType --- EtSum collection will not be saved...\n ";
      }
    }
    etMissResult->resize(1*bxCounter);
    htMissResult->resize(1*bxCounter);
    etTotResult->resize(1*bxCounter);
    etHadResult->resize(1*bxCounter);
  }

  bxCounter = 0;
  for(int itBX=HfSums->getFirstBX(); itBX<=HfSums->getLastBX(); ++itBX){

    if (itBX<bxMin_) continue;
    if (itBX>bxMax_) continue;

    bxCounter++;
    L1GctHFRingEtSums sum = L1GctHFRingEtSums::fromGctEmulator(itBX,
							       0,
							       0,
							       0,
							       0);
    for (CaloSpareBxCollection::const_iterator itCaloSpare = HfSums->begin(itBX);
	 itCaloSpare != HfSums->end(itBX); ++itCaloSpare){
      // if (CaloSpare::CaloSpareType::V2 == itCaloSpare->getType())
      // {
      // 	sum.setEtSum(3, itCaloSpare->hwPt());
      // } else if (CaloSpare::CaloSpareType::Centrality == itCaloSpare->getType())
      // {
      // 	sum.setEtSum(0, itCaloSpare->hwPt());
      // } else if (CaloSpare::CaloSpareType::Tau == itCaloSpare->getType())
      // {
      // 	sum.setEtSum(0, itCaloSpare->hwPt() & 0x7);
      // 	sum.setEtSum(1, (itCaloSpare->hwPt() >> 3) & 0x7);
      // 	sum.setEtSum(2, (itCaloSpare->hwPt() >> 6) & 0x7);
      // 	sum.setEtSum(3, (itCaloSpare->hwPt() >> 9) & 0x7);
      // }
      for(int i = 0; i < 4; i++)
      {
	sum.setEtSum(i, itCaloSpare->GetRing(i));
      }
    }
    hfRingEtSumResult->push_back(sum);

    hfRingEtSumResult->resize(1*bxCounter);
  }

  bxCounter = 0;
  for(int itBX=HfCounts->getFirstBX(); itBX<=HfCounts->getLastBX(); ++itBX){

    if (itBX<bxMin_) continue;
    if (itBX>bxMax_) continue;

    bxCounter++;
    L1GctHFBitCounts count = L1GctHFBitCounts::fromGctEmulator(itBX,
							       0,
							       0,
							       0,
							       0);
    for (CaloSpareBxCollection::const_iterator itCaloSpare = HfCounts->begin(itBX);
	 itCaloSpare != HfCounts->end(itBX); ++itCaloSpare){
      for(int i = 0; i < 4; i++)
      {
	count.setBitCount(i, itCaloSpare->GetRing(i));
      }
    }
    hfBitCountResult->push_back(count);
    hfBitCountResult->resize(1*bxCounter);
  }


  e.put(isoEmResult,"isoEm");
  e.put(nonIsoEmResult,"nonIsoEm");
  e.put(cenJetResult,"cenJets");
  e.put(forJetResult,"forJets");
  e.put(tauJetResult,"tauJets");
  e.put(isoTauJetResult,"isoTauJets");
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

// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void
L1TCaloUpgradeToGCTConverter::fillDescriptions(ConfigurationDescriptions& descriptions) {
  ParameterSetDescription desc;
  desc.add<int>("bxMin",0);
  desc.add<int>("bxMax",0);
  desc.add<edm::InputTag>("InputCollection",edm::InputTag("caloStage1Digis"));
  desc.add<edm::InputTag>("InputRlxTauCollection",edm::InputTag("caloStage1Digis:rlxTaus"));
  desc.add<edm::InputTag>("InputIsoTauCollection",edm::InputTag("caloStage1Digis:isoTaus"));
  desc.add<edm::InputTag>("InputHFSumsCollection",edm::InputTag("caloStage1Digis:HFRingSums"));
  desc.add<edm::InputTag>("InputHFCountsCollection",edm::InputTag("caloStage1Digis:HFBitCounts"));
  descriptions.add("L1TCaloUpgradeToGCTConverter", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TCaloUpgradeToGCTConverter);
