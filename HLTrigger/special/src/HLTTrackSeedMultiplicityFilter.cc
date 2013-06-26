#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTTrackSeedMultiplicityFilter : public HLTFilter {
public:
  explicit HLTTrackSeedMultiplicityFilter(const edm::ParameterSet&);
  ~HLTTrackSeedMultiplicityFilter();

private:
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

  edm::InputTag inputTag_;       // input tag identifying product containing track seeds
  unsigned int  min_seeds_;      // minimum number of track seeds
  unsigned int  max_seeds_;      // maximum number of track seeds

};

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

//
// constructors and destructor
//
 
HLTTrackSeedMultiplicityFilter::HLTTrackSeedMultiplicityFilter(const edm::ParameterSet& config) : HLTFilter(config),
  inputTag_  (config.getParameter<edm::InputTag>("inputTag")),
  min_seeds_ (config.getParameter<unsigned int>("minSeeds")),
  max_seeds_ (config.getParameter<unsigned int>("maxSeeds"))
{
  LogDebug("") << "Using the " << inputTag_ << " input collection";
  LogDebug("") << "Requesting at least " << min_seeds_ << " seeds";
  if(max_seeds_ > 0) 
    LogDebug("") << "...but no more than " << max_seeds_ << " seeds";
}

HLTTrackSeedMultiplicityFilter::~HLTTrackSeedMultiplicityFilter()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool HLTTrackSeedMultiplicityFilter::hltFilter(edm::Event& event, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputTag_);

  // get hold of products from Event
  edm::Handle<TrajectorySeedCollection> seedColl;
  event.getByLabel(inputTag_, seedColl);

  const TrajectorySeedCollection *rsSeedCollection = 0;


  if( seedColl.isValid() )
  {
    //std::cout << "Problem!!" << std::endl;
    rsSeedCollection = seedColl.product();
  } 
  else
  {
    return false;
  }


  // Number of trakc seeds in the collection

  unsigned int seedsize = rsSeedCollection->size();


  LogDebug("") << "Number of seeds: " << seedsize;

  // Apply the filter cuts

  bool accept = (seedsize >= min_seeds_);

  if(max_seeds_ > 0) 
    accept &= (seedsize <= max_seeds_);
  
  // return with final filter decision
  return accept;
}

// define as a framework module
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTTrackSeedMultiplicityFilter);
