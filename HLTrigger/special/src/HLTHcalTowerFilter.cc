/** \class HLTHcalTowerFilter
 *  
 *  This class is an EDFilter implementing the following requirement:
 *  the number of caltowers with hadEnergy>E_Thr less than N_Thr for HB/HE/HF sperately.
 *
 *  $Date: 2012/10/10 05:41:14 $
 *  $Revision: 1.8 $
 *
 *  \author Li Wenbo (PKU)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//
class HLTHcalTowerFilter : public HLTFilter 
{
public:
  explicit HLTHcalTowerFilter(const edm::ParameterSet &);
  ~HLTHcalTowerFilter();
  
private:
  virtual bool hltFilter(edm::Event &, const edm::EventSetup &, trigger::TriggerFilterObjectWithRefs & filterproduct);
  
  edm::InputTag inputTag_;    // input tag identifying product
  double min_E_HB_;           // energy threshold for HB in GeV
  double min_E_HE_;           // energy threshold for HE in GeV
  double min_E_HF_;           // energy threshold for HF in GeV
  int max_N_HB_;              // maximum number for HB
  int max_N_HE_;              // maximum number for HB
  int max_N_HF_;              // maximum number for HB
  int min_N_HF_;              // minimum number for HB
  int min_N_HFM_;             // minimum number for HB-
  int min_N_HFP_;             // minimum number for HB+
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
HLTHcalTowerFilter::HLTHcalTowerFilter(const edm::ParameterSet& config) : HLTFilter(config),
  inputTag_ (config.getParameter<edm::InputTag>("inputTag")),
  min_E_HB_ (config.getParameter<double>       ("MinE_HB")),
  min_E_HE_ (config.getParameter<double>       ("MinE_HE")),
  min_E_HF_ (config.getParameter<double>       ("MinE_HF")),
  max_N_HB_ (config.getParameter<int>          ("MaxN_HB")),
  max_N_HE_ (config.getParameter<int>          ("MaxN_HE")),
  max_N_HF_ (config.getParameter<int>          ("MaxN_HF")),
  min_N_HF_ (-1),
  min_N_HFM_ (-1),
  min_N_HFP_ (-1)
{

  if (config.existsAs<int>("MinN_HF")){
    min_N_HF_=config.getParameter<int>("MinN_HF");
  }
  if (config.existsAs<int>("MinN_HFM")){
    min_N_HFM_=config.getParameter<int>("MinN_HFM");
  }
  if (config.existsAs<int>("MinN_HFP")){
    min_N_HFP_=config.getParameter<int>("MinN_HFP");
  }

}

HLTHcalTowerFilter::~HLTHcalTowerFilter()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool 
HLTHcalTowerFilter::hltFilter(edm::Event& event, const edm::EventSetup& setup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
   
  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputTag_);

  // get hold of collection of objects
  Handle<CaloTowerCollection> towers;
  event.getByLabel(inputTag_, towers);

  // look at all objects, check cuts and add to filter object
  int n_HB = 0;
  int n_HE = 0;
  int n_HF = 0;
  int n_HFM = 0;
  int n_HFP = 0;
  double abseta = 0.0;
  double eta = 0.0;
  for(CaloTowerCollection::const_iterator i = towers->begin(); i != towers->end(); ++i) 
    {
      eta    = i->eta();
      abseta = std::abs(eta);
      if(abseta<1.305)
	{
	  if(i->hadEnergy() >= min_E_HB_) 
	    {
	      n_HB++;
	      //edm::Ref<CaloTowerCollection> ref(towers, std::distance(towers->begin(), i));
	      //filterproduct.addObject(TriggerJet, ref);
	    }
	}
      else if(abseta>=1.305 && abseta<3)
	{
	  if(i->hadEnergy() >= min_E_HE_)
	    {
	      n_HE++;
	      //edm::Ref<CaloTowerCollection> ref(towers, std::distance(towers->begin(), i));
	      //filterproduct.addObject(TriggerJet, ref);
	    }
	}
      else
	{
	  if(i->hadEnergy() >= min_E_HF_) 
	    {
	      n_HF++;
	      //edm::Ref<CaloTowerCollection> ref(towers, std::distance(towers->begin(), i));
	      //filterproduct.addObject(TriggerJet, ref);
	      if(eta >= 3) { n_HFP++; } else { n_HFM++; }
	    }
	}
    }

  // filter decision
  bool accept(n_HB<max_N_HB_ && n_HE<max_N_HE_ && n_HF<max_N_HF_ && n_HFP>=min_N_HFP_ && n_HFM>=min_N_HFM_ && n_HF>=min_N_HF_);

  return accept;
}

// define as a framework module
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTHcalTowerFilter);

