/** \class HLTSimpleJet
 *
 * See header file for documentation
 *
 *  $Date: 2006/06/17 00:18:35 $
 *  $Revision: 1.11 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTSimpleJet.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/RecoCandidate/interface/RecoCaloJetCandidate.h"

#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
HLTSimpleJet::HLTSimpleJet(const edm::ParameterSet& iConfig)
{
   module_ = iConfig.getParameter< std::string > ("input");
   ptcut_  = iConfig.getParameter<double> ("ptcut");
   njcut_  = iConfig.getParameter<int> ("njcut");

   LogDebug("") << "Input/ptcut/njcut : " << module_ << " " << ptcut_ << " " << njcut_;

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTSimpleJet::~HLTSimpleJet()
{
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
   for (; jet!=jets->end(); jet++) {
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

   return accept;
}
