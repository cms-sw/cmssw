/** \class HLTFiltCand
 *
 * See header file for documentation
 *
 *  $Date: 2006/06/16 18:55:56 $
 *  $Revision: 1.6 $
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

//
// constructors and destructor
//
 
HLTFiltCand::HLTFiltCand(const edm::ParameterSet& iConfig)
{
   using namespace reco;

   srcphot_ = iConfig.getParameter< std::string > ("srcPhot");
   srcelec_ = iConfig.getParameter< std::string > ("srcElec");
   srcmuon_ = iConfig.getParameter< std::string > ("srcMuon");
   srcjets_ = iConfig.getParameter< std::string > ("srcJets");

   pt_phot_ = iConfig.getParameter< double > ("ptPhot");
   pt_elec_ = iConfig.getParameter< double > ("ptElec");
   pt_muon_ = iConfig.getParameter< double > ("ptMuon");
   pt_jets_ = iConfig.getParameter< double > ("ptJets");

   // should use message logger instead of cout!
   std::cout << "HLTFiltCand created:" <<
     " g: " << srcphot_ << " " << pt_phot_ << 
     " e: " << srcelec_ << " " << pt_elec_ << 
     " m: " << srcmuon_ << " " << pt_muon_ << 
     " j: " << srcjets_ << " " << pt_jets_ << 
     std::endl;

   //register your products

   produces<reco::HLTFilterObjectWithRefs>();
}

HLTFiltCand::~HLTFiltCand()
{
   // should use message logger instead of cout!
   std::cout << "HLTFiltCand destroyed! " << std::endl;
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTFiltCand::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace reco;

   cout << "HLTFiltCand::filter start:" << endl;

   // All filter must create and fill a filter object
   // recording reconstructed physics objects making
   // satisfying (logical subexpressions of) this trigger

   // the filter object
   auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs);
   // ref to objects to be recorded
   edm::RefToBase<Candidate> ref;


   // Specific filter code

   // get hold of products from Event

   edm::Handle<PhotonCandidateCollection>      photons;
   edm::Handle<ElectronCandidateCollection>    electrons;
   edm::Handle<MuonCollection>                 muons;
   edm::Handle<RecoCaloJetCandidateCollection> jets;

   iEvent.getByLabel(srcphot_,photons  );
   iEvent.getByLabel(srcelec_,electrons);
   iEvent.getByLabel(srcmuon_,muons    );
   iEvent.getByLabel(srcjets_,jets     );


   // look for at least one g,e,m,j above its pt cut

   // photons
   bool         bphot(false);
   PhotonCandidateCollection::const_iterator aphot(photons->begin());
   PhotonCandidateCollection::const_iterator ophot(photons->end());
   PhotonCandidateCollection::const_iterator iphot;
   for (iphot=aphot; iphot!=ophot; iphot++) {
     if (iphot->pt() >= pt_phot_) {
       bphot=true;
       ref=edm::RefToBase<Candidate>(reco::PhotonCandidateRef(photons,distance(aphot,iphot)));
       filterproduct->putParticle(ref);
     }
   }

   // electrons
   bool         belec(false);
   ElectronCandidateCollection::const_iterator aelec(electrons->begin());
   ElectronCandidateCollection::const_iterator oelec(electrons->end());
   ElectronCandidateCollection::const_iterator ielec;
   for (ielec=aelec; ielec!=oelec; ielec++) {
     if (ielec->pt() >= pt_elec_) {
       belec=true;
       ref=edm::RefToBase<Candidate>(reco::ElectronCandidateRef(electrons,distance(aelec,ielec)));
       filterproduct->putParticle(ref);
     }
   }


   // muon
   bool         bmuon(false);
   MuonCollection::const_iterator amuon(muons->begin());
   MuonCollection::const_iterator omuon(muons->end());
   MuonCollection::const_iterator imuon;
   for (imuon=amuon; imuon!=omuon; imuon++) {
     if (imuon->pt() >= pt_muon_) {
       bmuon=true;
       ref=edm::RefToBase<Candidate>(reco::MuonRef(muons,distance(amuon,imuon)));
       filterproduct->putParticle(ref);
     }
   }

   // jets
   bool         bjets(false);
   RecoCaloJetCandidateCollection::const_iterator ajets(jets->begin());
   RecoCaloJetCandidateCollection::const_iterator ojets(jets->end());
   RecoCaloJetCandidateCollection::const_iterator ijets;
   for (ijets=ajets; ijets!=ojets; ijets++) {
     if (ijets->pt() >= pt_jets_) {
       bjets=true;
       ref=edm::RefToBase<Candidate>(reco::RecoCaloJetCandidateRef(jets,distance(ajets,ijets)));
       filterproduct->putParticle(ref);
     }
   }


   // final filter decision
   bool accept (bphot && belec && bmuon && bjets);

   // All filters: put filter object into the Event
   iEvent.put(filterproduct);

   std::cout << "HLTFiltCand::filter stop: " << std::endl;

   // return with final filter decision
   return accept;
}
