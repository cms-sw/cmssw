/** \class HLTFiltCand
 *
 * See header file for documentation
 *
 *  $Date: 2006/06/18 17:44:04 $
 *  $Revision: 1.8 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFiltCand.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/EgammaCandidates/interface/ElectronCandidate.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCandidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloJetCandidate.h"

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
   jetsTag_ = iConfig.getParameter< edm::InputTag > ("jetsTag");

   phot_pt_ = iConfig.getParameter< double > ("photPt");
   elec_pt_ = iConfig.getParameter< double > ("elecPt");
   muon_pt_ = iConfig.getParameter< double > ("muonPt");
   jets_pt_ = iConfig.getParameter< double > ("jetsPt");

   LogDebug("") << "created with:" <<
     " g: " << photTag_.encode() << " " << phot_pt_ << 
     " e: " << elecTag_.encode() << " " << elec_pt_ << 
     " m: " << muonTag_.encode() << " " << muon_pt_ << 
     " j: " << jetsTag_.encode() << " " << jets_pt_  ;

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

   // All filters must create and fill a filter object
   // recording any reconstructed physics objects 
   // satisfying this filter

   // The filter object
   auto_ptr<HLTFilterObjectWithRefs> 
     filterproduct (new HLTFilterObjectWithRefs(path(),module()));
   // Ref to Candidate objects to be recorded in filter object
   RefToBase<Candidate> ref;


   // Specific filter code

   // get hold of products from Event

   Handle<PhotonCandidateCollection>      photons;
   Handle<ElectronCandidateCollection>    electrons;
   Handle<MuonCollection>                 muons;
   Handle<RecoCaloJetCandidateCollection> jets;

   iEvent.getByLabel(photTag_,photons  );
   iEvent.getByLabel(elecTag_,electrons);
   iEvent.getByLabel(muonTag_,muons    );
   iEvent.getByLabel(jetsTag_,jets     );


   // look for at least one g,e,m,j above its pt cut

   // photons
   bool         bphot(false);
   PhotonCandidateCollection::const_iterator aphot(photons->begin());
   PhotonCandidateCollection::const_iterator ophot(photons->end());
   PhotonCandidateCollection::const_iterator iphot;
   for (iphot=aphot; iphot!=ophot; iphot++) {
     if (iphot->pt() >= phot_pt_) {
       bphot=true;
       ref=RefToBase<Candidate>(PhotonCandidateRef(photons,distance(aphot,iphot)));
       filterproduct->putParticle(ref);
     }
   }

   // electrons
   bool         belec(false);
   ElectronCandidateCollection::const_iterator aelec(electrons->begin());
   ElectronCandidateCollection::const_iterator oelec(electrons->end());
   ElectronCandidateCollection::const_iterator ielec;
   for (ielec=aelec; ielec!=oelec; ielec++) {
     if (ielec->pt() >= elec_pt_) {
       belec=true;
       ref=RefToBase<Candidate>(ElectronCandidateRef(electrons,distance(aelec,ielec)));
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

   // jets
   bool         bjets(false);
   RecoCaloJetCandidateCollection::const_iterator ajets(jets->begin());
   RecoCaloJetCandidateCollection::const_iterator ojets(jets->end());
   RecoCaloJetCandidateCollection::const_iterator ijets;
   for (ijets=ajets; ijets!=ojets; ijets++) {
     if (ijets->pt() >= jets_pt_) {
       bjets=true;
       ref=RefToBase<Candidate>(RecoCaloJetCandidateRef(jets,distance(ajets,ijets)));
       filterproduct->putParticle(ref);
     }
   }


   // final filter decision:
   bool accept (bphot && belec && bmuon && bjets);


   // All filters: put filter object into the Event
   iEvent.put(filterproduct);

   // return with final filter decision
   return accept;
}
