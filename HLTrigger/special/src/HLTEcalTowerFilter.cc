/** \class HLTEcalTowerFilter
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a
 *  single CaloTower requirement with an emEnergy threshold (not Et!)
 *
 *  $Date: 2012/01/21 15:00:16 $
 *  $Revision: 1.6 $
 *
 *  \author Seth Cooper
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTEcalTowerFilter : public HLTFilter {
public:
  explicit HLTEcalTowerFilter(const edm::ParameterSet &);
  ~HLTEcalTowerFilter();

private:
  virtual bool hltFilter(edm::Event &, const edm::EventSetup &, trigger::TriggerFilterObjectWithRefs & filterproduct);

  edm::InputTag inputTag_; // input tag identifying product
  double min_E_;           // energy threshold in GeV 
  double max_Eta_;         // maximum eta
  int min_N_;              // minimum number

};

#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

//
// constructors and destructor
//
HLTEcalTowerFilter::HLTEcalTowerFilter(const edm::ParameterSet& config) : HLTFilter(config),
  inputTag_ (config.getParameter<edm::InputTag>("inputTag")),
  min_E_    (config.getParameter<double>       ("MinE"   )),
  max_Eta_  (config.getParameter<double>       ("MaxEta"   )),
  min_N_    (config.getParameter<int>          ("MinN"   ))
{
  LogDebug("") << "Input/ecut/etacut/ncut : "
               << inputTag_.encode() << " "
               << min_E_ << " "
               << max_Eta_ << " "
               << min_N_ ;
}

HLTEcalTowerFilter::~HLTEcalTowerFilter()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
  bool 
HLTEcalTowerFilter::hltFilter(edm::Event& event, const edm::EventSetup& setup, trigger::TriggerFilterObjectWithRefs & filterproduct)
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

  // get hold of collection of objects
  Handle<CaloTowerCollection> towers;
  event.getByLabel(inputTag_, towers);

  LogDebug("HLTEcalTowerFilter") << "Number of towers: " << towers->size();

  // look at all objects, check cuts and add to filter object
  int n = 0;
  for (CaloTowerCollection::const_iterator i = towers->begin(); i != towers->end(); ++i) {
    if (i->emEnergy() >= min_E_ and fabs(i->eta()) <= max_Eta_) {
      ++n;
      //edm::Ref<CaloTowerCollection> ref(towers, std::distance(towers->begin(), i));
      //filterproduct.addObject(TriggerJet, ref);
    }
  }

  LogDebug("HLTEcalTowerFilter") << "Number of towers with eta < " << max_Eta_ << " and energy > " << min_E_ << ": " << n;

  // filter decision
  bool accept(n>=min_N_);

  return accept;
}

// define as a framework module
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTEcalTowerFilter);
