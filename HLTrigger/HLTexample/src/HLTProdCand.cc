/** \class HLTProdCand
 *
 * See header file for documentation
 *
 *  $Date: 2006/08/02 14:19:33 $
 *  $Revision: 1.3 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTexample/interface/HLTProdCand.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/METReco/interface/METCollection.h"

#include "CLHEP/HepMC/ReadHepMC.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
 
HLTProdCand::HLTProdCand(const edm::ParameterSet& iConfig)
{
   jetsTag_ = iConfig.getParameter< edm::InputTag > ("jetsTag");
   metsTag_ = iConfig.getParameter< edm::InputTag > ("metsTag");

   //register your products

   produces<reco::PhotonCollection>();
   produces<reco::ElectronCollection>();
   produces<reco::MuonCollection>();
   produces<reco::CaloJetCollection>();
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

   // produce dummy collections of photons, electrons, muons, jets, MET

   auto_ptr<PhotonCollection>   phot (new PhotonCollection);
   auto_ptr<ElectronCollection> elec (new ElectronCollection);
   auto_ptr<MuonCollection>     muon (new MuonCollection);
   auto_ptr<CaloJetCollection>  jets (new CaloJetCollection);
   auto_ptr<CaloMETCollection>  mets (new CaloMETCollection);

   // jets and MET are special: check if MC truths jets/mets is available
   edm::Handle<GenJetCollection> mcjets;
   edm::Handle<   METCollection> mcmets;
   bool foundJets(true);
   bool foundMets(true);
   try {
     iEvent.getByLabel(jetsTag_,mcjets);
   }
   catch(...) {
     foundJets=false;
   }
   LogDebug("") << "Found MC truth jets: " << foundJets;
   try {
     iEvent.getByLabel(metsTag_,mcmets);
   }
   catch(...) {
     foundMets=false;
   }
   LogDebug("") << "Found MC truth mets: " << foundMets;

   vector<edm::Handle<edm::HepMCProduct> > hepmcs;
   edm::Handle<edm::HepMCProduct> hepmc;
   iEvent.getManyByType(hepmcs);

   LogDebug("") << "Number of HepMC products found: " << hepmcs.size();

   math::XYZTLorentzVector p4;
   for (unsigned int i=0; i!=hepmcs.size(); i++) {
     hepmc=hepmcs[i];
     const HepMC::GenEvent* evt = hepmc->GetEvent();
     for (HepMC::GenEvent::particle_const_iterator pitr=evt->particles_begin(); pitr!=evt->particles_end(); pitr++) {
       if ( (*pitr)->status()==1) {
	 HepLorentzVector p = (*pitr)->momentum() ;
	 p4=math::XYZTLorentzVector(p.x(),p.y(),p.z(),p.t());
	 int ipdg = (*pitr)->pdg_id();
	 if (abs(ipdg)==11) {
	   elec->push_back(Electron(-ipdg/abs(ipdg),p4));
	 } else if (abs(ipdg)==13) {
	   muon->push_back(Muon(-ipdg/abs(ipdg),p4));
	 } else if (abs(ipdg)==22) {
	   phot->push_back(Photon(0,p4));
	 } else if (abs(ipdg)==12 || abs(ipdg)==14 || abs(ipdg)==16) {
	   if (!foundMets) {
	     SpecificCaloMETData specific;
	     mets->push_back(CaloMET(specific,p4.Et(),p4,math::XYZPoint(0,0,0)));
	   }
	 } else { 
	   if (!foundJets) {
	     CaloJet::Specific specific;
	     vector<CaloTowerDetId> ctdi(0);
	     jets->push_back(CaloJet(p4,specific,ctdi));
	   }
	 }
       }
     }
   }

   if (foundJets) {
     for (unsigned int i=0; i!=mcjets->size(); i++) {
       p4=((*mcjets)[i]).p4();
       CaloJet::Specific specific;
       vector<CaloTowerDetId> ctdi(0);
       jets->push_back(CaloJet(p4,specific,ctdi));
     }
   }

   if (foundMets) {
     for (unsigned int i=0; i!=mcmets->size(); i++) {
       p4=((*mcmets)[i]).p4();
       SpecificCaloMETData specific;
       mets->push_back(CaloMET(specific,p4.Et(),p4,math::XYZPoint(0,0,0)));
     }
   }

   LogTrace("") << "Number of g/e/m/j/M objects reconstructed: " 
        << phot->size() << " " 
        << elec->size() << " " 
        << muon->size() << " "
	<< jets->size() << " "
        << mets->size() ;

   // put them into the event

   iEvent.put(phot);
   iEvent.put(elec);
   iEvent.put(muon);
   iEvent.put(jets);
   iEvent.put(mets);

   return;
}
