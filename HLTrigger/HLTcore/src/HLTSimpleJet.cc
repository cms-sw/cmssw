/** \class HLTSimpleJet
 *
 * See header file for documentation
 *
 *  $Date: 2006/05/12 18:13:30 $
 *  $Revision: 1.9 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTSimpleJet.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/RecoCandidate/interface/RecoCaloJetCandidate.h"

#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

//
// constructors and destructor
//
HLTSimpleJet::HLTSimpleJet(const edm::ParameterSet& iConfig)
{
   module_ = iConfig.getParameter< std::string > ("input");
   ptcut_  = iConfig.getParameter<double> ("ptcut");
   njcut_  = iConfig.getParameter<int> ("njcut");

   // should use message logger instead of cout!
   std::cout << "HLTSimpleJet input: " << module_ << std::endl;
   std::cout << "             PTcut: " << ptcut_  << std::endl;
   std::cout << "    Number of jets: " << njcut_  << std::endl;

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTSimpleJet::~HLTSimpleJet()
{
   // should use message logger instead of cout!
   std::cout << "HLTSimpleJet destroyed! " << std::endl;
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTSimpleJet::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   using namespace reco;

   //   cout << "HLTSimpleJet::filter start:" << endl;

   // the filter object
   auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs);
   // ref to objects to be recorded
   edm::RefToBase<Candidate> ref;


   // get hold of jets
   Handle<RecoCaloJetCandidateCollection>  jets;
   iEvent.getByLabel (module_,jets);

   // look at all jets,  check cuts and add to filter object
   int n=0;
   RecoCaloJetCandidateCollection::const_iterator jet(jets->begin());
   for (; jet!=jets->end()&&n<njcut_; jet++) {
     //     cout << (*jet).pt() << endl;
     if ( (jet->pt()) >= ptcut_) {
       n++;
       ref=edm::RefToBase<Candidate>(reco::RecoCaloJetCandidateRef(jets,distance(jets->begin(),jet)));
       filterproduct->putParticle(ref);
     }
   }

   // filter decision
   bool accept(n>=njcut_);

   // put filter object into the Event
   iEvent.put(filterproduct);

   //   std::cout << "HLTSimpleJet::filter stop: " << n << std::endl;

   return accept;
}
