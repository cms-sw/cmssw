/** \class HLTProdCand
 *
 * See header file for documentation
 *
 *  $Date: 2006/08/11 07:02:55 $
 *  $Revision: 1.8 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTexample/interface/HLTProdCand.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/METReco/interface/GenMET.h"

#include "CLHEP/HepMC/ReadHepMC.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
   using namespace reco;

   // produce collections of photons, electrons, muons, taus, jets, MET

   auto_ptr<PhotonCollection>   phot (new PhotonCollection);
   auto_ptr<ElectronCollection> elec (new ElectronCollection);
   auto_ptr<MuonCollection>     muon (new MuonCollection);
   auto_ptr<CaloJetCollection>  taus (new CaloJetCollection); // stored as jets
   auto_ptr<CaloJetCollection>  jets (new CaloJetCollection);
   auto_ptr<CaloMETCollection>  mets (new CaloMETCollection);

   // photons, electrons, muons and taus: generator level
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
     vector<CaloTowerDetId> ctdi(0);
     jets->push_back(CaloJet(p4,specific,ctdi));
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
	 // 4-momentum
	 HepLorentzVector p((*pitr)->momentum()) ;
         math::XYZTLorentzVector 
	   p4(math::XYZTLorentzVector(p.x(),p.y(),p.z(),p.t()));

	 // particle type
	 int ipdg = (*pitr)->pdg_id();
	 if (abs(ipdg)==11) {
	   elec->push_back(Electron(-ipdg/abs(ipdg),p4));
	 } else if (abs(ipdg)==13) {
	   muon->push_back(Muon(-ipdg/abs(ipdg),p4));
	 } else if (abs(ipdg)==15) {
	   // taus
	   CaloJet::Specific specific;
	   vector<CaloTowerDetId> ctdi(0);
	   taus->push_back(CaloJet(p4,specific,ctdi));
	 } else if (abs(ipdg)==22) {
	   phot->push_back(Photon(0,p4));
	 } else if (abs(ipdg)==12 || abs(ipdg)==14 || abs(ipdg)==16) {
	   // neutrinos
	   if (nmets==-1) {
	     SpecificCaloMETData specific;
	     mets->push_back(CaloMET(specific,p4.Et(),p4,math::XYZPoint(0,0,0)));
	   }
	 } else {
	   // any other particle becomes a jet on its own (very crude)!
	   if (njets==-1) {
	     CaloJet::Specific specific;
	     vector<CaloTowerDetId> ctdi(0);
	     jets->push_back(CaloJet(p4,specific,ctdi));
	   }
	 }
       }
     }
   }

   LogTrace("") << "Number of g/e/m/t/j/M objects reconstructed:"
		<< " " << phot->size()
		<< " " << elec->size()
		<< " " << muon->size()
		<< " " << taus->size()
		<< " " << jets->size()
		<< " " << mets->size()
                ;

   // put them into the event

   iEvent.put(phot);
   iEvent.put(elec);
   iEvent.put(muon);
   iEvent.put(taus,"taus");
   iEvent.put(jets,"jets");
   iEvent.put(mets);

   return;
}
