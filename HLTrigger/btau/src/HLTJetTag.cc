/** \class HLTJetTag
 *
  * See header file for documentation
 *
 *  $Date: 2007/10/07 09:41:33 $
 *  $Revision: 1.6 $
 *
 *  \author Arnaud Gay, Ian Tomalin
 *
 */

#include "HLTrigger/btau/interface/HLTJetTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

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
  produces<trigger::TriggerFilterObjectWithRefs>();
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
  auto_ptr<trigger::TriggerFilterObjectWithRefs>
    filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));

  Handle<JetTagCollection> jetsHandle;
  iEvent.getByLabel(jetTag_, jetsHandle);

  const JetTagCollection & jets = *(jetsHandle.product());

  // Look at all jets in decreasing order of Et.
  int nJet = 0;
  int nTag = 0;
  for (JetTagCollection::const_iterator jet = jets.begin(); jet != jets.end(); jet++) {
    LogTrace("") << "Jet " << nJet
                 << " : Et = " << jet->first->et()
                 << " , tag value = " << jet->second;
    ++nJet;
    // Check if jet is tagged.
    if ( (min_Tag_ <= jet->second) and (jet->second <= max_Tag_) ) {
      ++nTag;

      // Store a reference to the jets which passed tagging cuts
      // N.B. this *should* work as we start from a CaloJet in HLT
      filterproduct->addObject(trigger::TriggerBJet, jet->first.castTo<reco::CaloJetRef>() );
    }
  }

  // filter decision
  bool accept = (nTag >= min_N_);

  // put filter object into the Event
  iEvent.put(filterproduct);

  edm::LogInfo("") << label_ << " trigger accept ? = " << accept
                   << " nTag/nJet = " << nTag << "/" << nJet << std::endl;

  return accept;
}
