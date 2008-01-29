/** \class HLTJetTag
 *
  * See header file for documentation
 *
 *  $Date: 2007/05/10 13:12:40 $
 *  $Revision: 1.4 $
 *
 *  \author Arnaud Gay, Ian Tomalin
 *
 */

#include "HLTrigger/btau/interface/HLTJetTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
HLTJetTag::HLTJetTag(const edm::ParameterSet& iConfig) :
  jetTag_  (iConfig.getParameter<edm::InputTag> ("JetTag")),
  min_Tag_ (iConfig.getParameter<double>        ("MinTag")),
  max_Tag_ (iConfig.getParameter<double>        ("MaxTag")),
  min_N_   (iConfig.getParameter<int>           ("MinN")),
  label_   (iConfig.getParameter<std::string>   ("@module_label"))
{

  edm::LogInfo("") << " TRIGGER CUTS: " << label_ << std::endl
                    << " Type of tagged jets used: " << jetTag_.encode() << std::endl
    		<< " Min/Max tag value [" << min_Tag_ << "--" << max_Tag_ << "]" << std::endl
    		<< " Min no. tagged jets = " << min_N_ << std::endl;

  //register your products
  produces<reco::HLTFilterObjectWithRefs>();
}

HLTJetTag::~HLTJetTag()
{
  edm::LogInfo("") << "Destroyed !";
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

  //  edm::LogInfo("") << "Start at event="<<iEvent.id().event();

  // Needed to store jets used for triggering.
  auto_ptr<HLTFilterObjectWithRefs>
    filterproduct (new HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  RefToBase<Candidate> ref;

  Handle<JetTagCollection> jetsHandle;
  iEvent.getByLabel(jetTag_, jetsHandle);

  const JetTagCollection & jets = *(jetsHandle.product());

  // Look at all jets in decreasing order of Et.
  int nJet(0);
  int nTag(0);
  JetTagCollection::const_iterator jet;
  for (jet = jets.begin(); jet != jets.end(); jet++) {
    edm::LogInfo("") << "Jet " << nJet
	   	     << " : Et = " << jet->jet()->et()
		     << " , No. of tracks = " << jet->tracks().size()
		     << " , tag value = " << jet->discriminator();
    nJet++;
    //  Check if jet is tagged.
    if ( (min_Tag_ <= jet->discriminator()) && 
	 (jet->discriminator() <= max_Tag_) ) {
      nTag++;

      // Store (ref to) jets which passed tagging cuts
      ref=CandidateBaseRef(jet->jet() );
      filterproduct->putParticle(ref);
      if (nTag==1) { // also store ProductID of Product containing AssociationMap
	ProductID pid( (jet->jet()).id() );
	filterproduct->putPID(pid);
      }
    }
  }

  // filter decision
  bool accept (nTag >= min_N_);

  // put filter object into the Event
  iEvent.put(filterproduct);

  edm::LogInfo("") <<  label_ << " trigger accept ? = " << accept
	           << " nTag/nJet = " << nTag << "/" << nJet << std::endl;

  return accept;
}
