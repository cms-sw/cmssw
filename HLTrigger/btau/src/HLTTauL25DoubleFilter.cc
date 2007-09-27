/** \class HLTTauL25DoubleFilter
 *
 * See header file for documentation
 *
 *  $Date: 2007/05/14 14:19:08 $
 *  $Revision: 1.4 $
 *
 *  \author S. Gennai
 *
 */

#include "HLTrigger/btau/interface/HLTTauL25DoubleFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
HLTTauL25DoubleFilter::HLTTauL25DoubleFilter(const edm::ParameterSet& iConfig) :
  HLTTauL25DoubleFilterLabel_ (iConfig.getParameter<std::string>   ("@module_label")),
  JetTag_         (iConfig.getParameter<edm::InputTag> ("JetTag")),
  L1Code_          (iConfig.getParameter<edm::InputTag> ("L1Code"))
{


  //register your products
  produces<reco::HLTFilterObjectWithRefs>();
}

HLTTauL25DoubleFilter::~HLTTauL25DoubleFilter()
{
  LogDebug("") << "Destroyed!";
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTTauL25DoubleFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;

  LogDebug("") << "Start at event="<<iEvent.id().event();

  // Needed to store jets used for triggering.
  auto_ptr<HLTFilterObjectWithRefs>
    filterproduct (new HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  RefToBase<Candidate> ref;

  Handle<JetTagCollection> jetsHandle;
  iEvent.getByLabel(JetTag_, jetsHandle);
  Handle< vector<int> > l1Decision;
  iEvent.getByLabel(L1Code_,l1Decision);


  int l1Deco = -1000;
    l1Deco = *(l1Decision->begin());

  const JetTagCollection & jets = *(jetsHandle.product());

  // Look at all jets in decreasing order of Et.
  int nJet(0);
  int nTag(0);
  JetTagCollection::const_iterator jet;
  for (jet = jets.begin(); jet != jets.end(); jet++) {
    LogTrace("") << "Jet " << nJet
		 << " : Et = " << jet->jet()->et()
		 << " , No. of tracks = " << jet->tracks().size()
		 << " , tag value = " << jet->discriminator();
    nJet++;
    //  Check if jet is tagged.
    if ( jet->discriminator() == 1 ) 
      nTag++;
    
    // Store (ref to) jets which passed tagging cuts
    ref=CandidateBaseRef( jet->jet() );
      filterproduct->putParticle(ref);
      if (nTag==1) { // also store ProductID of Product containing AssociationMap
	ProductID pid( jet->jet().id() );
	filterproduct->putPID(pid);
      }
	
  }

  // filter decision see L2TauJetsProvider for coding description;
  bool accept = false;
  //  if(l1Deco == -100) cout <<"ERROR"<<endl;
  if(l1Deco ==  0) accept = false;
  if(l1Deco == 1 && nTag > 0) accept = true; 
  if(l1Deco == 2 && nTag > 1) accept = true;
  if(l1Deco == 3 ) accept = true;
  //  cout <<"L25SingleTauMixing decision "<<accept<<endl;



  //mixed case is missing
  if(l1Deco == 6) accept = true;
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  LogDebug("") <<  HLTTauL25DoubleFilterLabel_ << " accept = " << accept
	       << " nTag/nJet = " << nTag << "/" << nJet;
  
  return accept;
}
