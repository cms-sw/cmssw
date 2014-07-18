/** \class HLTHtMhtFilter
 *
 * See header file for documentation
 *
 *  \author Steven Lowette
 *
 */

#include "HLTrigger/JetMET/interface/HLTHtMhtFilter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"


// Constructor
HLTHtMhtFilter::HLTHtMhtFilter(const edm::ParameterSet & iConfig) : HLTFilter(iConfig),
  minHt_     ( iConfig.getParameter<std::vector<double> >("minHt") ),
  minMht_    ( iConfig.getParameter<std::vector<double> >("minMht") ),
  minMeff_   ( iConfig.getParameter<std::vector<double> >("minMeff") ),
  meffSlope_ ( iConfig.getParameter<std::vector<double> >("meffSlope") ),
  htLabels_  ( iConfig.getParameter<std::vector<edm::InputTag> >("htLabels") ),
  mhtLabels_ ( iConfig.getParameter<std::vector<edm::InputTag> >("mhtLabels") ),
  nOrs_      ( htLabels_.size() ) {  // number of settings to .OR.
    if (!( htLabels_.size() == minHt_.size() &&
           htLabels_.size() == minMht_.size() &&
           htLabels_.size() == minMeff_.size() &&
           htLabels_.size() == meffSlope_.size() &&
           htLabels_.size() == mhtLabels_.size() ) ||
        htLabels_.size() == 0 ) {
        nOrs_ = (minHt_.size()     < nOrs_ ? minHt_.size()     : nOrs_);
        nOrs_ = (minMht_.size()    < nOrs_ ? minMht_.size()    : nOrs_);
        nOrs_ = (minMeff_.size()   < nOrs_ ? minMeff_.size()   : nOrs_);
        nOrs_ = (meffSlope_.size() < nOrs_ ? meffSlope_.size() : nOrs_);
        nOrs_ = (mhtLabels_.size() < nOrs_ ? mhtLabels_.size() : nOrs_);
        edm::LogError("HLTHtMhtFilter") << "inconsistent module configuration!";
    }

    for(unsigned int i=0; i<nOrs_; ++i) {
        m_theHtToken.push_back(consumes<reco::METCollection>(htLabels_[i]));
        m_theMhtToken.push_back(consumes<reco::METCollection>(mhtLabels_[i]));
    }

}

// Destructor
HLTHtMhtFilter::~HLTHtMhtFilter() {}

// Fill descriptions
void HLTHtMhtFilter::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    std::vector<edm::InputTag> tmp1(1, edm::InputTag("hltHtMhtProducer"));
    std::vector<double>        tmp2(1, 0.);
    edm::ParameterSetDescription desc;
    makeHLTFilterDescription(desc);
    desc.add<std::vector<edm::InputTag> >("htLabels",  tmp1);
    desc.add<std::vector<edm::InputTag> >("mhtLabels", tmp1);
    tmp2[0] = 250; desc.add<std::vector<double> >("minHt",     tmp2);
    tmp2[0] =  70; desc.add<std::vector<double> >("minMht",    tmp2);
    tmp2[0] =   0; desc.add<std::vector<double> >("minMeff",   tmp2);
    tmp2[0] =   1; desc.add<std::vector<double> >("meffSlope", tmp2);
    descriptions.add("hltHtMhtFilter", desc);
}

// Make filter decision
bool HLTHtMhtFilter::hltFilter(edm::Event & iEvent, const edm::EventSetup & iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const {

    bool accept = false;

    // Take the .OR. of all sets of requirements
    for (unsigned int i = 0; i < nOrs_; ++i) {
      // Create the reference to the output filter objects
      if (saveTags()) {
	filterproduct.addCollectionTag(htLabels_[i]);
	filterproduct.addCollectionTag(mhtLabels_[i]);
      }

      edm::Handle<reco::METCollection> hht;
      iEvent.getByToken(m_theHtToken[i], hht);
      double ht = 0;
      if (hht->size() > 0)  ht = hht->front().sumEt();
      
      edm::Handle<reco::METCollection> hmht;
      iEvent.getByToken(m_theMhtToken[i], hmht);
      double mht = 0;
      if (hmht->size() > 0)  mht = hmht->front().pt();
      
      // Check if the event passes this cut set
      accept = accept || (ht > minHt_[i] && mht > minMht_[i] && sqrt(mht + meffSlope_[i]*ht) > minMeff_[i]);
      // In principle we could break if accepted, but in order to save
      // for offline analysis all possible decisions we keep looping here
      // in term of timing this will not matter much; typically 1 or 2 cut-sets
      // will be checked only
      
      // Store the object that was cut on and the ref to it
      // (even if it is not accepted)
      edm::Ref<reco::METCollection> htref(hht,0);
      edm::Ref<reco::METCollection> mhtref(hmht,0);
      filterproduct.addObject(trigger::TriggerTHT, htref);  // save as TriggerTHT object
      filterproduct.addObject(trigger::TriggerMHT, mhtref);  // save as TriggerMHT object
    }

    return accept;
}
