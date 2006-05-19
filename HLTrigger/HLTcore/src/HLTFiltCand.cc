/** \class HLTFiltCand
 *
 * See header file for documentation
 *
 *  $Date: 2006/05/18 16:57:25 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFiltCand.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/EgammaCandidates/interface/ElectronCandidate.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCandidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

//
// constructors and destructor
//
 
HLTFiltCand::HLTFiltCand(const edm::ParameterSet& iConfig)
{
   using namespace reco;

   module_ = iConfig.getParameter< std::string > ("input");

   // should use message logger instead of cout!
   std::cout << "HLTFiltCand created: " << module_ << std::endl;

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

   edm::Handle<PhotonCandidateCollection>   photons;
   edm::Handle<ElectronCandidateCollection> electrons;
   edm::Handle<MuonCollection>     muons;

   iEvent.getByLabel(module_,photons);
   iEvent.getByLabel(module_,electrons);
   iEvent.getByLabel(module_,muons);

   // create filter object
   auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs);

   // dummy "gem" filter - pass the event with the 3rd gamma, 6th electron, 9th muon
   //                      and record these as having fired the trigger!

   auto_ptr<edm::RefToBase<Candidate> > ref;

   ref=edm::makeRefToBase<Candidate>(reco::PhotonCandidateRef(photons, 3));
   filterproduct->putParticle(ref);

   ref=edm::makeRefToBase<Candidate>(reco::ElectronCandidateRef(electrons, 6));
   filterproduct->putParticle(ref);

   ref=edm::makeRefToBase<Candidate>(reco::MuonRef(muons, 9));
   filterproduct->putParticle(ref);

   bool accept(true);
   filterproduct->setAccept(accept);

   // put filter object into the Event
   iEvent.put(filterproduct);

   std::cout << "HLTFiltCand::filter stop: " << std::endl;

   return accept;
}
