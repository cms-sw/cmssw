/** \class HLTProdCand
 *
 * See header file for documentation
 *
 *  $Date: 2007/04/13 16:06:16 $
 *  $Revision: 1.37 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTexample/interface/HLTProdCand.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/METReco/interface/GenMET.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HepPDT/ParticleID.hh"

//
// constructors and destructor
//
 
HLTProdCand::HLTProdCand(const edm::ParameterSet& iConfig) :
  jetsTag_ (iConfig.getParameter<edm::InputTag>("jetsTag")),
  metsTag_ (iConfig.getParameter<edm::InputTag>("metsTag"))
{
  LogDebug("") << "Inputs: jets/mets: " << jetsTag_.encode() << " / " << metsTag_.encode();

   //register your products

   produces<reco::PhotonCollection>();
   produces<reco::ElectronCollection>();
   produces<reco::MuonCollection>();
   produces<reco::CaloJetCollection>("taus");
   produces<reco::CaloJetCollection>("jets");
   produces<reco::CaloMETCollection>();
   produces<reco::RecoChargedCandidateCollection>();
   produces<reco::RecoEcalCandidateCollection>();

}

HLTProdCand::~HLTProdCand()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
HLTProdCand::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;

   // produce collections of photons, electrons, muons, taus, jets, MET
   auto_ptr<PhotonCollection>   phot (new PhotonCollection);
   auto_ptr<ElectronCollection> elec (new ElectronCollection);
   auto_ptr<MuonCollection>     muon (new MuonCollection);
   auto_ptr<CaloJetCollection>  taus (new CaloJetCollection); // stored as jets
   auto_ptr<CaloJetCollection>  jets (new CaloJetCollection);
   auto_ptr<CaloMETCollection>  mets (new CaloMETCollection);

   // as well as charged tracks and elmg. superclusters
   auto_ptr<RecoChargedCandidateCollection> trck (new RecoChargedCandidateCollection);
   auto_ptr<RecoEcalCandidateCollection>    ecal (new RecoEcalCandidateCollection);

   // jets and MET are special: check whether clustered jets and mets
   // exist already
   int njets(-1);
   edm::Handle<GenJetCollection> mcjets;
   try {iEvent.getByLabel(jetsTag_,mcjets);} catch (...) {;}
   if (mcjets.isValid()) njets=mcjets->size();
   LogDebug("") << "MC-truth jets found: " << njets;
   for (int i=0; i<njets; i++) {
     math::XYZTLorentzVector p4(((*mcjets)[i]).p4());
     CaloJet::Specific specific;
     Jet::Constituents jetconst;
     jets->push_back(CaloJet(p4,specific,jetconst));
   }

   int nmets(-1);
   edm::Handle<GenMETCollection> mcmets;
   try {iEvent.getByLabel(metsTag_,mcmets);} catch(...) {;}
   if (mcmets.isValid()) nmets=mcmets->size();
   LogDebug("") << "MC-truth mets found: " << nmets;
   for (int i=0; i<nmets; i++) {
     math::XYZTLorentzVector p4(((*mcmets)[i]).p4());
     SpecificCaloMETData specific;
     mets->push_back(CaloMET(specific,p4.Et(),p4,math::XYZPoint(0,0,0)));
   }

   // photons, electrons, muons and taus: generator level
   // tracks: all charged particles; superclusters: electrons and photons

   // get hold of generator records
   vector<edm::Handle<edm::HepMCProduct> > hepmcs;
   edm::Handle<edm::HepMCProduct> hepmc;
   iEvent.getManyByType(hepmcs);
   LogDebug("") << "Number of HepMC products found: " << hepmcs.size();

   // loop over all final-state particles in all generator records
   for (unsigned int i=0; i!=hepmcs.size(); i++) {
     hepmc=hepmcs[i];
     const HepMC::GenEvent* evt = hepmc->GetEvent();
     for (HepMC::GenEvent::particle_const_iterator pitr=evt->particles_begin(); pitr!=evt->particles_end(); pitr++) {

       // stable particles only!
       if ( (*pitr)->status()==1) {
	 // particle type
	 const int ipdg((*pitr)->pdg_id());
	 // 4-momentum
	 const HepMC::FourVector p((*pitr)->momentum());
         const math::XYZTLorentzVector 
	   p4(math::XYZTLorentzVector(p.x(),p.y(),p.z(),p.t()));
	 // charge
	 const int qX3(HepPDT::ParticleID(ipdg).threeCharge());

	 // charged particles yield tracks
	 if ( qX3 != 0 ) {
	   trck->push_back(RecoChargedCandidate(qX3/abs(qX3),p4));
	 }

	 if (abs(ipdg)==11) {
	   // e+ e-
	   elec->push_back(    Electron     (-ipdg/abs(ipdg),p4));
	   ecal->push_back(RecoEcalCandidate(-ipdg/abs(ipdg),p4));
	 } else if (abs(ipdg)==13) {
	   // mu+ mu-
	   muon->push_back(Muon(-ipdg/abs(ipdg),p4));
	 } else if (abs(ipdg)==15 || abs(ipdg)==17) {
	   // tau+ tau- or 4th generation tau'+ tau'-
	   CaloJet::Specific specific;
	   Jet::Constituents jetconst;
	   taus->push_back(CaloJet(p4,specific,jetconst));
	 } else if (abs(ipdg)==22) {
	   // photon
	   phot->push_back(    Photon(0,p4,math::XYZPoint(0,0,0),1.0,1.0,p4.E()));
	   ecal->push_back(RecoEcalCandidate(0,p4));
	 } else if (abs(ipdg)==12 || abs(ipdg)==14 || abs(ipdg)==16 || abs(ipdg)==18) {
	   // neutrinos (e mu tau 4th generation)
	   if (nmets==-1) {
	     // if no prepared mets, each becomes a met on its own (crude)!
	     SpecificCaloMETData specific;
	     mets->push_back(CaloMET(specific,p4.Et(),p4,math::XYZPoint(0,0,0)));
	   }
	 } else {
	   // any other particle
	   if (njets==-1) {
	     // if no prepared jets, each becomes a jet on its own (crude)!
	     CaloJet::Specific specific;
	     Jet::Constituents jetconst;
	     jets->push_back(CaloJet(p4,specific,jetconst));
	   }
	 }
       }
     }
   }

   LogDebug("") << "Number of g/e/m/t/j/M/SC/TR objects reconstructed:"
		<< " " << phot->size()
		<< " " << elec->size()
		<< " " << muon->size()
		<< " " << taus->size()
		<< " " << jets->size()
		<< " " << mets->size()
		<< " " << ecal->size()
		<< " " << trck->size()
                ;

   // put them into the event

   iEvent.put(phot);
   iEvent.put(elec);
   iEvent.put(muon);
   iEvent.put(taus,"taus");
   iEvent.put(jets,"jets");
   iEvent.put(mets);
   iEvent.put(ecal);
   iEvent.put(trck);

   return;
}
