/** \class HLTFiltCand
 *
 * See header file for documentation
 *
 *  $Date: 2012/01/21 14:56:59 $
 *  $Revision: 1.18 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTfilters/interface/HLTFiltCand.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
 
HLTFiltCand::HLTFiltCand(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  photTag_ (iConfig.getParameter<edm::InputTag>("photTag")),
  elecTag_ (iConfig.getParameter<edm::InputTag>("elecTag")),
  muonTag_ (iConfig.getParameter<edm::InputTag>("muonTag")),
  tausTag_ (iConfig.getParameter<edm::InputTag>("tausTag")),
  jetsTag_ (iConfig.getParameter<edm::InputTag>("jetsTag")),
  metsTag_ (iConfig.getParameter<edm::InputTag>("metsTag")),
  mhtsTag_ (iConfig.getParameter<edm::InputTag>("mhtsTag")),
  trckTag_ (iConfig.getParameter<edm::InputTag>("trckTag")),
  ecalTag_ (iConfig.getParameter<edm::InputTag>("ecalTag")),
  min_Pt_  (iConfig.getParameter<double>("MinPt"))
{
  LogDebug("") << "MinPt cut " << min_Pt_
   << " g: " << photTag_.encode()
   << " e: " << elecTag_.encode()
   << " m: " << muonTag_.encode()
   << " t: " << tausTag_.encode()
   << " j: " << jetsTag_.encode()
   << " M: " << metsTag_.encode()
   << " H: " << mhtsTag_.encode()
   <<" TR: " << trckTag_.encode()
   <<" SC: " << ecalTag_.encode()
   ;
}

HLTFiltCand::~HLTFiltCand()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTFiltCand::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // The filter object
   if (saveTags()) {
     filterproduct.addCollectionTag(photTag_);
     filterproduct.addCollectionTag(elecTag_);
     filterproduct.addCollectionTag(muonTag_);
     filterproduct.addCollectionTag(tausTag_);
     filterproduct.addCollectionTag(jetsTag_);
     filterproduct.addCollectionTag(metsTag_);
     filterproduct.addCollectionTag(mhtsTag_);
     filterproduct.addCollectionTag(trckTag_);
     filterproduct.addCollectionTag(ecalTag_);
   }

   // Specific filter code

   // get hold of products from Event

   Handle<RecoEcalCandidateCollection>   photons;
   Handle<ElectronCollection> electrons;
   Handle<RecoChargedCandidateCollection>     muons;
   Handle<CaloJetCollection>  taus;
   Handle<CaloJetCollection>  jets;
   Handle<CaloMETCollection>  mets;
   Handle<METCollection>      mhts;
   Handle<RecoChargedCandidateCollection> trcks;
   Handle<RecoEcalCandidateCollection>    ecals;

   iEvent.getByLabel(photTag_,photons  );
   iEvent.getByLabel(elecTag_,electrons);
   iEvent.getByLabel(muonTag_,muons    );
   iEvent.getByLabel(tausTag_,taus     );
   iEvent.getByLabel(jetsTag_,jets     );
   iEvent.getByLabel(metsTag_,mets     );
   iEvent.getByLabel(mhtsTag_,mhts     );
   iEvent.getByLabel(trckTag_,trcks    );
   iEvent.getByLabel(ecalTag_,ecals    );


   // look for at least one g,e,m,t,j,M,H,TR,SC above its pt cut

   // photons
   int nphot(0);
   RecoEcalCandidateCollection::const_iterator aphot(photons->begin());
   RecoEcalCandidateCollection::const_iterator ophot(photons->end());
   RecoEcalCandidateCollection::const_iterator iphot;
   for (iphot=aphot; iphot!=ophot; iphot++) {
     if (iphot->pt() >= min_Pt_) {
       nphot++;
       RecoEcalCandidateRef ref(RecoEcalCandidateRef(photons,distance(aphot,iphot)));
       filterproduct.addObject(TriggerPhoton,ref);
     }
   }

   // electrons
   int nelec(0);
   ElectronCollection::const_iterator aelec(electrons->begin());
   ElectronCollection::const_iterator oelec(electrons->end());
   ElectronCollection::const_iterator ielec;
   for (ielec=aelec; ielec!=oelec; ielec++) {
     if (ielec->pt() >= min_Pt_) {
       nelec++;
       ElectronRef ref(ElectronRef(electrons,distance(aelec,ielec)));
       filterproduct.addObject(-TriggerElectron,ref);
     }
   }

   // muon
   int nmuon(0);
   RecoChargedCandidateCollection::const_iterator amuon(muons->begin());
   RecoChargedCandidateCollection::const_iterator omuon(muons->end());
   RecoChargedCandidateCollection::const_iterator imuon;
   for (imuon=amuon; imuon!=omuon; imuon++) {
     if (imuon->pt() >= min_Pt_) {
       nmuon++;
       RecoChargedCandidateRef ref(RecoChargedCandidateRef(muons,distance(amuon,imuon)));
       filterproduct.addObject(TriggerMuon,ref);
     }
   }

   // taus (are stored as jets)
   int ntaus(0);
   CaloJetCollection::const_iterator ataus(taus->begin());
   CaloJetCollection::const_iterator otaus(taus->end());
   CaloJetCollection::const_iterator itaus;
   for (itaus=ataus; itaus!=otaus; itaus++) {
     if (itaus->pt() >= min_Pt_) {
       ntaus++;
       CaloJetRef ref(CaloJetRef(taus,distance(ataus,itaus)));
       filterproduct.addObject(-TriggerTau,ref);
     }
   }

   // jets
   int njets(0);
   CaloJetCollection::const_iterator ajets(jets->begin());
   CaloJetCollection::const_iterator ojets(jets->end());
   CaloJetCollection::const_iterator ijets;
   for (ijets=ajets; ijets!=ojets; ijets++) {
     if (ijets->pt() >= min_Pt_) {
       njets++;
       CaloJetRef ref(CaloJetRef(jets,distance(ajets,ijets)));
       filterproduct.addObject(TriggerJet,ref);
     }
   }

   // mets
   int nmets(0);
   CaloMETCollection::const_iterator amets(mets->begin());
   CaloMETCollection::const_iterator omets(mets->end());
   CaloMETCollection::const_iterator imets;
   for (imets=amets; imets!=omets; imets++) {
     if (imets->pt() >= min_Pt_) {
       nmets++;
       CaloMETRef ref(CaloMETRef(mets,distance(amets,imets)));
       filterproduct.addObject(TriggerMET,ref);
     }
   }

   // mhts
   int nmhts(0);
   METCollection::const_iterator amhts(mhts->begin());
   METCollection::const_iterator omhts(mhts->end());
   METCollection::const_iterator imhts;
   for (imhts=amhts; imhts!=omhts; imhts++) {
     if (imhts->pt() >= min_Pt_) {
       nmhts++;
       METRef ref(METRef(mhts,distance(amhts,imhts)));
       filterproduct.addObject(TriggerMHT,ref);
     }
   }

   // trcks
   int ntrck(0);
   RecoChargedCandidateCollection::const_iterator atrcks(trcks->begin());
   RecoChargedCandidateCollection::const_iterator otrcks(trcks->end());
   RecoChargedCandidateCollection::const_iterator itrcks;
   for (itrcks=atrcks; itrcks!=otrcks; itrcks++) {
     if (itrcks->pt() >= min_Pt_) {
       ntrck++;
       RecoChargedCandidateRef ref(RecoChargedCandidateRef(trcks,distance(atrcks,itrcks)));
       filterproduct.addObject(TriggerTrack,ref);
     }
   }

   // ecals
   int necal(0);
   RecoEcalCandidateCollection::const_iterator aecals(ecals->begin());
   RecoEcalCandidateCollection::const_iterator oecals(ecals->end());
   RecoEcalCandidateCollection::const_iterator iecals;
   for (iecals=aecals; iecals!=oecals; iecals++) {
     if (iecals->pt() >= min_Pt_) {
       necal++;
       RecoEcalCandidateRef ref(RecoEcalCandidateRef(ecals,distance(aecals,iecals)));
       filterproduct.addObject(TriggerCluster,ref);
     }
   }

   // error case
   // filterproduct.addObject(0,Ref<vector<int> >());

   // final filter decision:
   const bool accept ( (nphot>0) && (nelec>0) && (nmuon>0) && (ntaus>0) &&
		       //   (njets>0) && (nmets>0) && (nmhts>=0) && (ntrck>0) && (necal>0) );
		       (njets>0) && (nmets>0) && (ntrck>0) && (necal>0) );

   LogDebug("") << "Number of g/e/m/t/j/M/H/TR/SC objects accepted:"
		<< " " << nphot
		<< " " << nelec
		<< " " << nmuon
		<< " " << ntaus
		<< " " << njets
		<< " " << nmets
		<< " " << nmhts
		<< " " << ntrck
		<< " " << necal
                ;

   // return with final filter decision
   return accept;
}
