/** \class HLT1CaloJetEnergy
 *
 * See header file for documentation
 *
 *  $Date: 2011/05/01 14:41:36 $
 *  $Revision: 1.11 $
 *
 *  \author Jim Brooke
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTfilters/interface/HLT1CaloJetEnergy.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

//
// constructors and destructor
//
HLT1CaloJetEnergy::HLT1CaloJetEnergy(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  inputTag_ (iConfig.getParameter<edm::InputTag>("inputTag")),
  min_E_    (iConfig.getParameter<double>       ("MinE"   )),
  max_Eta_  (iConfig.getParameter<double>       ("MaxEta"   )),
  min_N_    (iConfig.getParameter<int>          ("MinN"   ))
{
   LogDebug("") << "Input/ecut/etacut/ncut : "
		<< inputTag_.encode() << " "
		<< min_E_ << " "
		<< max_Eta_ << " "
		<< min_N_ ;
}

HLT1CaloJetEnergy::~HLT1CaloJetEnergy()
{
}

void
HLT1CaloJetEnergy::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltStoppedHSCPIterativeCone5CaloJets"));
  desc.add<bool>("saveTags",false);
  desc.add<double>("MinE",20.0);
  desc.add<double>("MaxEta",3.0);
  desc.add<int>("MinN",1);
  descriptions.add("hlt1CaloJetEnergy",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool 
HLT1CaloJetEnergy::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // The filter object
   if (saveTags()) filterproduct.addCollectionTag(inputTag_);

   // Ref to Candidate object to be recorded in filter object
   Ref<CaloJetCollection> ref;


   // get hold of collection of objects
   Handle<CaloJetCollection> jets;
   iEvent.getByLabel (inputTag_,jets);

   // look at all objects, check cuts and add to filter object
   int n(0);
   CaloJetCollection::const_iterator i ( jets->begin() );
   for (; i!=jets->end(); i++) {
     if ( i->energy() >= min_E_  &&
	  fabs(i->eta()) <= max_Eta_   ) {
       n++;
       ref=Ref<CaloJetCollection>(jets,distance(jets->begin(),i));
       filterproduct.addObject(TriggerJet,ref);
     }
   }

   // filter decision
   bool accept(n>=min_N_);

   return accept;
}


#include "FWCore/Framework/interface/MakerMacros.h"
