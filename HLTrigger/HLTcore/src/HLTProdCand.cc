/** \class HLTProdCand
 *
 * See header file for documentation
 *
 *  $Date: 2006/05/19 17:26:43 $
 *  $Revision: 1.2 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTProdCand.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/EgammaCandidates/interface/ElectronCandidate.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCandidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"

//
// constructors and destructor
//
 
HLTProdCand::HLTProdCand(const edm::ParameterSet& iConfig)
{
   using namespace reco;

   n_ = iConfig.getParameter< unsigned int > ("n");

   // should use message logger instead of cout!
   std::cout << "HLTProdCand created: " << n_ << std::endl;

   //register your products

   produces<reco::PhotonCandidateCollection>();
   produces<reco::ElectronCandidateCollection>();
   produces<reco::MuonCollection>();

}

HLTProdCand::~HLTProdCand()
{
   // should use message logger instead of cout!
   std::cout << "HLTProdCand destroyed! " << std::endl;
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

   cout << "HLTProdCand::produce start:" << endl;

   // produce dummy collections of photons, electrons, muons

   auto_ptr<PhotonCandidateCollection>   phot (new PhotonCandidateCollection);
   auto_ptr<ElectronCandidateCollection> elec (new ElectronCandidateCollection);
   auto_ptr<MuonCollection>     muon (new MuonCollection);

   // fill collections with fake data

   for (unsigned int i=0; i!=n_; i++) {
     math::XYZTLorentzVector p4(100.0*i,200.0*i,200.0*i,300.0*i);
     phot->push_back(  PhotonCandidate(0,p4));
     elec->push_back(ElectronCandidate(1,p4));
     muon->push_back(            Muon(-1,p4));
   }

   // put them into the event

   iEvent.put(phot);
   iEvent.put(elec);
   iEvent.put(muon);

   cout << "HLTProdCand::produce stop:" << endl;
}
