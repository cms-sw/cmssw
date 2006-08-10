/** \class HLTFiltCand
 *
 * See header file for documentation
 *
 *  $Date: 2006/08/10 09:25:35 $
 *  $Revision: 1.7 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTexample/interface/HLTFiltCand.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
 
HLTFiltCand::HLTFiltCand(const edm::ParameterSet& iConfig)
{

   photTag_ = iConfig.getParameter< edm::InputTag > ("photTag");
   elecTag_ = iConfig.getParameter< edm::InputTag > ("elecTag");
   muonTag_ = iConfig.getParameter< edm::InputTag > ("muonTag");
   tausTag_ = iConfig.getParameter< edm::InputTag > ("tausTag");
   jetsTag_ = iConfig.getParameter< edm::InputTag > ("jetsTag");
   metsTag_ = iConfig.getParameter< edm::InputTag > ("metsTag");

   phot_pt_ = iConfig.getParameter< double > ("photPt");
   elec_pt_ = iConfig.getParameter< double > ("elecPt");
   muon_pt_ = iConfig.getParameter< double > ("muonPt");
   taus_pt_ = iConfig.getParameter< double > ("tausPt");
   jets_pt_ = iConfig.getParameter< double > ("jetsPt");
   mets_pt_ = iConfig.getParameter< double > ("metsPt");

   LogDebug("") << "created with:" <<
     " g: " << photTag_.encode() << " " << phot_pt_ << 
     " e: " << elecTag_.encode() << " " << elec_pt_ << 
     " m: " << muonTag_.encode() << " " << muon_pt_ << 
     " t: " << tausTag_.encode() << " " << taus_pt_ << 
     " j: " << jetsTag_.encode() << " " << jets_pt_ <<
     " M: " << metsTag_.encode() << " " << mets_pt_ ;

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();

}

HLTFiltCand::~HLTFiltCand()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTFiltCand::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // The filter object
   auto_ptr<HLTFilterObjectWithRefs> 
     filterproduct (new HLTFilterObjectWithRefs(path(),module()));
   // Ref to Candidate objects to be recorded in filter object
   RefToBase<Candidate> ref;


   // Specific filter code

   // get hold of products from Event

   Handle<PhotonCollection>   photons;
   Handle<ElectronCollection> electrons;
   Handle<MuonCollection>     muons;
   Handle<CaloJetCollection>  taus;
   Handle<CaloJetCollection>  jets;
   Handle<CaloMETCollection>  mets;

   iEvent.getByLabel(photTag_,photons  );
   iEvent.getByLabel(elecTag_,electrons);
   iEvent.getByLabel(muonTag_,muons    );
   iEvent.getByLabel(tausTag_,taus     );
   iEvent.getByLabel(jetsTag_,jets     );
   iEvent.getByLabel(metsTag_,mets     );


   // look for at least one g,e,m,t,j,M above its pt cut

   // photons
   bool         bphot(false);
   PhotonCollection::const_iterator aphot(photons->begin());
   PhotonCollection::const_iterator ophot(photons->end());
   PhotonCollection::const_iterator iphot;
   for (iphot=aphot; iphot!=ophot; iphot++) {
     if (iphot->pt() >= phot_pt_) {
       bphot=true;
       ref=RefToBase<Candidate>(PhotonRef(photons,distance(aphot,iphot)));
       filterproduct->putParticle(ref);
     }
   }

   // electrons
   bool         belec(false);
   ElectronCollection::const_iterator aelec(electrons->begin());
   ElectronCollection::const_iterator oelec(electrons->end());
   ElectronCollection::const_iterator ielec;
   for (ielec=aelec; ielec!=oelec; ielec++) {
     if (ielec->pt() >= elec_pt_) {
       belec=true;
       ref=RefToBase<Candidate>(ElectronRef(electrons,distance(aelec,ielec)));
       filterproduct->putParticle(ref);
     }
   }

   // muon
   bool         bmuon(false);
   MuonCollection::const_iterator amuon(muons->begin());
   MuonCollection::const_iterator omuon(muons->end());
   MuonCollection::const_iterator imuon;
   for (imuon=amuon; imuon!=omuon; imuon++) {
     if (imuon->pt() >= muon_pt_) {
       bmuon=true;
       ref=RefToBase<Candidate>(MuonRef(muons,distance(amuon,imuon)));
       filterproduct->putParticle(ref);
     }
   }

   // taus (are stored as jets)
   bool         btaus(false);
   CaloJetCollection::const_iterator ataus(taus->begin());
   CaloJetCollection::const_iterator otaus(taus->end());
   CaloJetCollection::const_iterator itaus;
   for (itaus=ataus; itaus!=otaus; itaus++) {
     if (itaus->pt() >= taus_pt_) {
       btaus=true;
       ref=RefToBase<Candidate>(CaloJetRef(taus,distance(ataus,itaus)));
       filterproduct->putParticle(ref);
     }
   }

   // jets
   bool         bjets(false);
   CaloJetCollection::const_iterator ajets(jets->begin());
   CaloJetCollection::const_iterator ojets(jets->end());
   CaloJetCollection::const_iterator ijets;
   for (ijets=ajets; ijets!=ojets; ijets++) {
     if (ijets->pt() >= jets_pt_) {
       bjets=true;
       ref=RefToBase<Candidate>(CaloJetRef(jets,distance(ajets,ijets)));
       filterproduct->putParticle(ref);
     }
   }

   // mets
   bool         bmets(false);
   CaloMETCollection::const_iterator amets(mets->begin());
   CaloMETCollection::const_iterator omets(mets->end());
   CaloMETCollection::const_iterator imets;
   for (imets=amets; imets!=omets; imets++) {
     if (imets->pt() >= mets_pt_) {
       bmets=true;
       ref=RefToBase<Candidate>(CaloMETRef(mets,distance(amets,imets)));
       filterproduct->putParticle(ref);
     }
   }


   // final filter decision:
   bool accept (bphot && belec && bmuon && btaus && bjets && bmets);


   // All filters: put filter object into the Event
   iEvent.put(filterproduct);

   // return with final filter decision
   return accept;
}
