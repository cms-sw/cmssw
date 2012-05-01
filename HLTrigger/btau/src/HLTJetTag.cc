/** \class HLTJetTag
 *
 *  This class is an HLTFilter (a spcialized EDFilter) implementing 
 *  tagged multi-jet trigger for b and tau. 
 *  It should be run after the normal multi-jet trigger.
 *
 *  $Date: 2008/04/23 11:57:53 $
 *  $Revision: 1.9 $
 *
 *  \author Arnaud Gay, Ian Tomalin
 *  \maintainer Andrea Bocci
 *
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "HLTJetTag.h"

//
// constructors and destructor
//

HLTJetTag::HLTJetTag(const edm::ParameterSet & config) :
  m_jetTag(  config.getParameter<edm::InputTag> ("JetTag") ),
  m_minTag(  config.getParameter<double>        ("MinTag") ),
  m_maxTag(  config.getParameter<double>        ("MaxTag") ),
  m_minJets( config.getParameter<int>           ("MinJets") ),
  m_saveTags( config.getParameter<bool>          ("saveTags") ),
  m_label(   config.getParameter<std::string>   ("@module_label") )
{

  edm::LogInfo("") << m_label << " (HLTJetTag) trigger cuts: " << std::endl
                   << "\ttype of tagged jets used: " << m_jetTag.encode() << std::endl
                   << "\tmin/max tag value: [" << m_minTag << ".." << m_maxTag << "]" << std::endl
                   << "\tmin no. tagged jets: " << m_minJets << std::endl;

  //register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTJetTag::~HLTJetTag()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTJetTag::filter(edm::Event& event, const edm::EventSetup& setup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;

  // store jets used for triggering.
  auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));

  edm::Handle<JetTagCollection> h_jetTag;
  event.getByLabel(m_jetTag, h_jetTag);
  const reco::JetTagCollection & jetTags = * h_jetTag;

  if (m_saveTags and jetTags.size()) {
    // find out which InputTag identifies the jets being tagged, and request it to be stored in the event
    // if there are no tagged jets, there is nothing to save
    ProductID jetsId = jetTags.begin()->first.id();
    edm::Handle<edm::View<reco::Jet> > h_jets;
    event.get(jetsId, h_jets);
    const edm::Provenance & jets_data = * h_jets.provenance();
    edm::InputTag jets_name( jets_data.moduleLabel(), jets_data.productInstanceName(), jets_data.processName() );

    filterproduct->addCollectionTag( jets_name );
  }

  // Look at all jets in decreasing order of Et.
  int nJet = 0;
  int nTag = 0;
  for (JetTagCollection::const_iterator jet = jetTags.begin(); jet != jetTags.end(); ++jet) {
    LogTrace("") << "Jet " << nJet
                 << " : Et = " << jet->first->et()
                 << " , tag value = " << jet->second;
    ++nJet;
    // Check if jet is tagged.
    if ( (m_minTag <= jet->second) and (jet->second <= m_maxTag) ) {
      ++nTag;

      // Store a reference to the jets which passed tagging cuts
      // N.B. this *should* work as we start from a CaloJet in HLT
      filterproduct->addObject(trigger::TriggerBJet, jet->first.castTo<reco::CaloJetRef>() );
    }
  }

  // filter decision
  bool accept = (nTag >= m_minJets);

  // put filter object into the Event
  event.put(filterproduct);

  edm::LogInfo("") << m_label << " trigger accept ? = " << accept
                   << " nTag/nJet = " << nTag << "/" << nJet << std::endl;

  return accept;
}
