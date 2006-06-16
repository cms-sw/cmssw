/** \class HLTFiltCand
 *
 * See header file for documentation
 *
 *  $Date: 2006/05/25 16:49:34 $
 *  $Revision: 1.5 $
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

   // get hold of products from Event

   edm::Handle<PhotonCandidateCollection>      photons;
   edm::Handle<ElectronCandidateCollection>    electrons;
   edm::Handle<MuonCollection>                 muons;
   edm::Handle<RecoCaloJetCandidateCollection> jets;

   iEvent.getByLabel(srcphot_,photons  );
   iEvent.getByLabel(srcelec_,electrons);
   iEvent.getByLabel(srcmuon_,muons    );
   iEvent.getByLabel(srcjets_,jets     );

   edm::RefToBase<Candidate> ref;

   // create filter object
   auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs);

   // look for at least one g,e,m,j above its pt cut

   // photons
   bool         bphot(false);
   PhotonCandidateCollection::const_iterator aphot(photons->begin());
   PhotonCandidateCollection::const_iterator ophot(photons->end());
   PhotonCandidateCollection::const_iterator iphot;
   for (iphot=aphot; (iphot!=ophot)&&(!bphot); iphot++) {
     if (iphot->pt() >= pt_phot_) {
       bphot=true;
       ref=edm::RefToBase<Candidate>(reco::PhotonCandidateRef(photons,distance(aphot,iphot)));
       filterproduct->putParticle(ref);
       // at this point ref has released and is no longer valid!
     }
   }

   // electrons
   bool         belec(false);
   ElectronCandidateCollection::const_iterator aelec(electrons->begin());
   ElectronCandidateCollection::const_iterator oelec(electrons->end());
   ElectronCandidateCollection::const_iterator ielec;
   for (ielec=aelec; (ielec!=oelec)&&(!belec); ielec++) {
     if (ielec->pt() >= pt_elec_) {
       belec=true;
       ref=edm::RefToBase<Candidate>(reco::ElectronCandidateRef(electrons,distance(aelec,ielec)));
       filterproduct->putParticle(ref);
       // at this point ref has released and is no longer valid!
     }
   }


   // muon
   bool         bmuon(false);
   MuonCollection::const_iterator amuon(muons->begin());
   MuonCollection::const_iterator omuon(muons->end());
   MuonCollection::const_iterator imuon;
   for (imuon=amuon; (imuon!=omuon)&&(!bmuon); imuon++) {
     if (imuon->pt() >= pt_muon_) {
       bmuon=true;
       ref=edm::RefToBase<Candidate>(reco::MuonRef(muons,distance(amuon,imuon)));
       filterproduct->putParticle(ref);
       // at this point ref has released and is no longer valid!
     }
   }

   // jets
   bool         bjets(false);
   RecoCaloJetCandidateCollection::const_iterator ajets(jets->begin());
   RecoCaloJetCandidateCollection::const_iterator ojets(jets->end());
   RecoCaloJetCandidateCollection::const_iterator ijets;
   for (ijets=ajets; (ijets!=ojets)&&(!bjets); ijets++) {
     if (ijets->pt() >= pt_jets_) {
       bjets=true;
       ref=edm::RefToBase<Candidate>(reco::RecoCaloJetCandidateRef(jets,distance(ajets,ijets)));
       filterproduct->putParticle(ref);
       // at this point ref has released and is no longer valid!
     }
   }

   // final decision
   bool accept (bphot && belec && bmuon && bjets);
   filterproduct->setAccept(accept);

   // put filter object into the Event
   iEvent.put(filterproduct);

   std::cout << "HLTFiltCand::filter stop: " << std::endl;

   return accept;
}
