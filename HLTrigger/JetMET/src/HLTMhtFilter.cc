/** \class HLTMhtFilter
 *
 * See header file for documentation
 *
 *  \author Steven Lowette
 *
 */

#include "HLTrigger/JetMET/interface/HLTMhtFilter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"


// Constructor
HLTMhtFilter::HLTMhtFilter(const edm::ParameterSet & iConfig) : HLTFilter(iConfig),
  minMht_    ( iConfig.getParameter<std::vector<double> >("minMht") ),
  mhtLabels_ ( iConfig.getParameter<std::vector<edm::InputTag> >("mhtLabels") ),
  nOrs_      ( mhtLabels_.size() ) {  // number of settings to .OR.
    if (!(mhtLabels_.size() == minMht_.size()) ||
        mhtLabels_.size() == 0 ) {
        nOrs_ = (minMht_.size()    < nOrs_ ? minMht_.size()    : nOrs_);
        edm::LogError("HLTMhtFilter") << "inconsistent module configuration!";
    }

    for(unsigned int i=0; i<nOrs_; ++i) {
        m_theMhtToken.push_back(consumes<reco::METCollection>(mhtLabels_[i]));
    }

}

// Destructor
HLTMhtFilter::~HLTMhtFilter() {}

// Fill descriptions
void HLTMhtFilter::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    std::vector<edm::InputTag> tmp1(1, edm::InputTag("hltMhtProducer"));
    std::vector<double>        tmp2(1, 0.);
    edm::ParameterSetDescription desc;
    makeHLTFilterDescription(desc);
    desc.add<std::vector<edm::InputTag> >("mhtLabels", tmp1);
    tmp2[0] =  70; desc.add<std::vector<double> >("minMht", tmp2);
    descriptions.add("hltMhtFilter", desc);
}

// Make filter decision
bool HLTMhtFilter::hltFilter(edm::Event & iEvent, const edm::EventSetup & iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const {

    reco::METRef mhtref;

    bool accept = false;

    // Take the .OR. of all sets of requirements
    for (unsigned int i = 0; i < nOrs_; ++i) {

      // Create the reference to the output filter objects
      if (saveTags())  filterproduct.addCollectionTag(mhtLabels_[i]);

      edm::Handle<reco::METCollection> hmht;
      iEvent.getByToken(m_theMhtToken[i], hmht);
      double mht = 0;
      if (hmht->size() > 0)  mht = hmht->front().pt();
      
      // Check if the event passes this cut set
      accept = accept || (mht > minMht_[i]);
      // In principle we could break if accepted, but in order to save
      // for offline analysis all possible decisions we keep looping here
      // in term of timing this will not matter much; typically 1 or 2 cut-sets
      // will be checked only
      
      // Store the ref (even if it is not accepted)
      mhtref = reco::METRef(hmht, 0);
      
      filterproduct.addObject(trigger::TriggerMHT, mhtref);  // save as TriggerMHT object
    }
    
    return accept;
}
