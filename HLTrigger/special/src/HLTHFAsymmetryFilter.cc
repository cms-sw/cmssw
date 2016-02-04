#include "HLTrigger/special/interface/HLTHFAsymmetryFilter.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HLTHFAsymmetryFilter::HLTHFAsymmetryFilter(const edm::ParameterSet& iConfig)
{
  HFHits_   = iConfig.getParameter<edm::InputTag>("HFHitCollection");  
  eCut_HF_  = iConfig.getParameter<double>("ECut_HF");
  os_asym_ = iConfig.getParameter<double>("OS_Asym_max");
  ss_asym_ = iConfig.getParameter<double>("SS_Asym_min");

  //register your products

  produces<trigger::TriggerFilterObjectWithRefs>();
}


HLTHFAsymmetryFilter::~HLTHFAsymmetryFilter()
{
 

}


// ------------ method called to produce the data  ------------
bool
HLTHFAsymmetryFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<HFRecHitCollection> HFRecHitsH;
  
  iEvent.getByLabel(HFHits_,HFRecHitsH);

  double asym_hf_1  = -1;
  double asym_hf_2  = -1;

  int n_hf[2]   = {0,0};
  double e_hf[2] = {0.,0.};

  //Select interesting HFRecHits
  for (HFRecHitCollection::const_iterator it=HFRecHitsH->begin(); it!=HFRecHitsH->end(); it++) {
    if (it->energy()>eCut_HF_) 
    {
      if (it->id().zside() == -1)
      {
	++n_hf[0];
	e_hf[0] += it->energy();
      }
      else
      {
	++n_hf[1];
	e_hf[1] += it->energy();
      }
    }
  }
 

  for (int i=0;i<2;++i) 
  {
    if (n_hf[i])
      e_hf[i] /= n_hf[i];
  }

  if (e_hf[0]+e_hf[1] != 0)
  {
    asym_hf_1 = e_hf[0]/(e_hf[0]+e_hf[1]);
    asym_hf_2 = e_hf[1]/(e_hf[0]+e_hf[1]);
  }
  else 
  {
    return false;
  }

  bool pkam_1 = (asym_hf_1 >= ss_asym_ || asym_hf_1 <= os_asym_);
  bool pkam_2 = (asym_hf_2 >= ss_asym_ || asym_hf_2 <= os_asym_);

  if (pkam_1 || pkam_2)
    return true;

  return false;
}

