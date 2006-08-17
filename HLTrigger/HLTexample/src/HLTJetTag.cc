/** \class HLTJetTag
 *
 * See header file for documentation
 *
 *  $Date: 2006/08/17 07:23:36 $
 *  $Revision: 1.1 $
 *
 *  \author Arnaud Gay, Ian Tomalin
 *
 */

#include "HLTrigger/HLTexample/interface/HLTJetTag.h"

#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
HLTJetTag::HLTJetTag(const edm::ParameterSet& iConfig) :
  JetTag_ (iConfig.getParameter<edm::InputTag>("JetTag")),
  Min_Tag_(iConfig.getParameter<double>("MinTag")),
  Max_Tag_(iConfig.getParameter<double>("MaxTag")),
  Min_N_  (iConfig.getParameter<int>("MinN"))
{
  LogDebug ("") << "Input and cuts: " << JetTag_.encode() 
		<< " Tag value [" << Min_Tag_ << " " << Max_Tag_ << "]"
		<< " MinN = " << Min_N_;

  //register your products
  produces<reco::HLTFilterObjectWithRefs>();
}

HLTJetTag::~HLTJetTag()
{
  LogDebug ("") << "Destroyed!";
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTJetTag::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
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

  const JetTagCollection & jets = *(jetsHandle.product());

  // Look at all jets in decreasing order of Et.
  int nJet(0);
  int nTag(0);
  JetTagCollection::const_iterator jet;
  for (jet = jets.begin(); jet != jets.end(); jet++) {
    LogTrace("") << "Jet " << nJet
		 << " : Et = " << jet->jet().et()
		 << " , No. of tracks = " << jet->tracks().size()
		 << " , tag value = " << jet->discriminator();
    nJet++;
    //  Check if jet is tagged.
    if ( (Min_Tag_ <= jet->discriminator()) && 
	 (jet->discriminator() <= Max_Tag_) ) {
      nTag++;
      // Store jets which passed tagging cuts

      // Need to construct a Ref to the Jet
      // from the "jet" used as key of the association map
      // ref = RefToBase<Candidate>(...);
      // filterproduct->putParticle(ref);
    }
  }

  // filter decision
  bool accept (nTag >= Min_N_);

  // put filter object into the Event
  iEvent.put(filterproduct);

  LogDebug("") << "accept = " << accept
	       << " nTag/nJet = " << nTag << "/" << nJet;

  return accept;
}
