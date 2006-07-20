/** \class HLTProdCand
 *
 * See header file for documentation
 *
 *  $Date: 2006/07/03 06:26:06 $
 *  $Revision: 1.11 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTProdCand.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"

#include "CLHEP/HepMC/ReadHepMC.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
 
HLTProdCand::HLTProdCand(const edm::ParameterSet& iConfig)
{
   using namespace reco;

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


   vector<edm::Handle<edm::HepMCProduct> > hepmcs;
   edm::Handle<edm::HepMCProduct> hepmc;
   iEvent.getManyByType(hepmcs);

   LogDebug("") << "Number of HepMC products found: " << hepmcs.size();

   ParticleKinematics p4;
   if (hepmcs.size()>0) {
     hepmc=hepmcs[0];
     const HepMC::GenEvent* evt = hepmc->GetEvent();
     for (HepMC::GenEvent::vertex_const_iterator vitr=evt->vertices_begin(); vitr!= evt->vertices_end(); vitr++) {
       for (HepMC::GenVertex::particle_iterator pitr=(*vitr)->particles_begin(HepMC::children);
                                               pitr!=(*vitr)->particles_end(HepMC::children); pitr++) {
	 if ( (*pitr)->status()==1) {
	   HepLorentzVector p = (*pitr)->momentum() ;
           p4=ParticleKinematics(math::XYZTLorentzVector(p.x(),p.y(),p.z(),p.t()));
           int ipdg = (*pitr)->pdg_id();
           if (abs(ipdg)==11) {
             elec->push_back(Electron(0,p4));
	   } else if (abs(ipdg)==13) {
             muon->push_back(Muon(0,p4));
           } else if (abs(ipdg)==22) {
             phot->push_back(Photon(0,p4));
           } else if (abs(ipdg)==12 || abs(ipdg)==14 || abs(ipdg)==16) {
	     SpecificCaloMETData specific;
             mets->push_back(CaloMET(specific,p4.et(),p4,Point(0,0,0)));
	   } else { 
	     CaloJet::Specific specific;
             vector<CaloTowerDetId> ctdi(0);
             jets->push_back(CaloJet(p4,specific,ctdi));
	   }
         }
       }
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
