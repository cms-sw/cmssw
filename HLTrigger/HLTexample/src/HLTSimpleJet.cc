/** \class HLTSimpleJet
 *
 * See header file for documentation
 *
 *  $Date: 2006/06/28 01:41:22 $
 *  $Revision: 1.2 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTexample/interface/HLTSimpleJet.h"

#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
HLTSimpleJet::HLTSimpleJet(const edm::ParameterSet& iConfig)
{
   inputTag_ = iConfig.getParameter< edm::InputTag > ("inputTag");
   ptcut_  = iConfig.getParameter<double> ("ptcut");
   njcut_  = iConfig.getParameter<int> ("njcut");

   LogDebug("") << "Input/ptcut/njcut : " << inputTag_.encode() << " " << ptcut_ << " " << njcut_;

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
   using namespace std;
   using namespace edm;
   using namespace reco;

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // The filter object
   auto_ptr<HLTFilterObjectWithRefs>
     filterproduct (new HLTFilterObjectWithRefs(path(),module()));
   // Ref to Candidate object to be recorded in filter object
   RefToBase<Candidate> ref;


   // get hold of jets
   Handle<CaloJetCollection> jets;
   iEvent.getByLabel (inputTag_,jets);

   // look at all jets,  check cuts and add to filter object
   int n(0);
   CaloJetCollection::const_iterator jet(jets->begin());
   for (; jet!=jets->end(); jet++) {
     if ( (jet->pt()) >= ptcut_) {
       n++;
       ref=RefToBase<Candidate>(CaloJetRef(jets,distance(jets->begin(),jet)));
       filterproduct->putParticle(ref);
     }
   }

   // filter decision
   bool accept(n>=njcut_);

   // put filter object into the Event
   iEvent.put(filterproduct);

   return accept;
}
