/** \class HLTHcalTowerFilter
 *  
 *  This class is an EDFilter which requires 
 *  the number of caltowers with E(had)>5Gev less than a certain value
 *
 *  \author Li Wenbo (PKU)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTHcalTowerFilter : public HLTFilter {
public:
   explicit HLTHcalTowerFilter(const edm::ParameterSet &);
   ~HLTHcalTowerFilter();

private:
   virtual bool filter(edm::Event &, const edm::EventSetup &);

   edm::InputTag inputTag_; // input tag identifying product
   bool saveTag_;           // whether to save this tag
   double min_E_;           // energy threshold in GeV 
   //   double max_Eta_;         // maximum eta
   int max_N_;              // maximum number

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
HLTHcalTowerFilter::HLTHcalTowerFilter(const edm::ParameterSet& config) :
   inputTag_ (config.getParameter<edm::InputTag>("inputTag")),
   saveTag_  (config.getUntrackedParameter<bool>("saveTag", false)),
   min_E_    (config.getParameter<double>       ("MinE"   )),
   //   max_Eta_  (config.getParameter<double>       ("MaxEta"   )),
   max_N_    (config.getParameter<int>          ("MaxN"   ))
{
   // register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTHcalTowerFilter::~HLTHcalTowerFilter()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool 
HLTHcalTowerFilter::filter(edm::Event& event, const edm::EventSetup& setup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;

   // The filter object
   std::auto_ptr<TriggerFilterObjectWithRefs> filterobject (new TriggerFilterObjectWithRefs(path(),module()));
   if (saveTag_) filterobject->addCollectionTag(inputTag_);

   // get hold of collection of objects
   Handle<CaloTowerCollection> towers;
   event.getByLabel(inputTag_, towers);

   // look at all objects, check cuts and add to filter object
   int n = 0;
   for(CaloTowerCollection::const_iterator i = towers->begin(); i != towers->end(); ++i) {
      if(i->hadEnergy() >= min_E_) {
      n++;
      //edm::Ref<CaloTowerCollection> ref(towers, std::distance(towers->begin(), i));
      //filterobject->addObject(TriggerJet, ref);
      }
   }

   // filter decision
   bool accept(n<=max_N_);

   // put filter object into the Event
   event.put(filterobject);

   return accept;
}

// define as a framework module
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTHcalTowerFilter);

