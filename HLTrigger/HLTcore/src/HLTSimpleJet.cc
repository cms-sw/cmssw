/** \class HLTSimpleJet
 *
 * See header file for documentation
 *
 *  $Date: 2006/04/26 09:27:44 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTSimpleJet.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

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

   //   cout << "HLTSimpleJet::filter start:" << endl;

   // get hold of jets
   Handle<CaloJetCollection>  jets;
   iEvent.getByLabel (module_,jets);

   // create filter object
   auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs);

   // look at all jets,  check cuts and add to filter object
   int n=0;
   CaloJetCollection::const_iterator jet(jets->begin());
   for (; jet!=jets->end()&&n<njcut_; jet++) {
     //     cout << (*jet).pt() << endl;
     if ( (jet->pt()) >= ptcut_) {
       n++;
       filterproduct->putJet(Ref<CaloJetCollection>(jets,distance(jets->begin(),jet)));
     }
   }

   // filter decision
   bool accept(n>=njcut_);
   filterproduct->setAccept(accept);
   // put filter object into the Event
   iEvent.put(filterproduct);

   //   std::cout << "HLTSimpleJet::filter stop: " << n << std::endl;

   return accept;
}
