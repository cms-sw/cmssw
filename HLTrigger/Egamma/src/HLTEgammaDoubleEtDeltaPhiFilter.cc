/** \class HLTEgammaDoubleEtDeltaPhiFilter
 *  This filter will require only two HLT photons and 
 *  DeltaPhi between the two photons larger than 2.5
 * 
 *  \author Li Wenbo (PKU)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaDoubleEtDeltaPhiFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

//
// constructors and destructor
//
HLTEgammaDoubleEtDeltaPhiFilter::HLTEgammaDoubleEtDeltaPhiFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) 
{
   inputTag_ = iConfig.getParameter< edm::InputTag > ("inputTag");
   etcut_  = iConfig.getParameter<double> ("etcut");
   minDeltaPhi_ =   iConfig.getParameter<double> ("minDeltaPhi");
   relaxed_ = iConfig.getUntrackedParameter<bool> ("relaxed",true) ;
   L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand"); 
   L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand");
}

HLTEgammaDoubleEtDeltaPhiFilter::~HLTEgammaDoubleEtDeltaPhiFilter(){}

// ------------ method called to produce the data  ------------
bool
HLTEgammaDoubleEtDeltaPhiFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
   using namespace trigger;
   // The filter object
   if (saveTags()) {
     filterproduct.addCollectionTag(L1IsoCollTag_);
     if (relaxed_) filterproduct.addCollectionTag(L1NonIsoCollTag_);
   }

   // get hold of filtered candidates
   edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
   iEvent.getByLabel (inputTag_,PrevFilterOutput);
  
   std::vector<edm::Ref<reco::RecoEcalCandidateCollection> >  recoecalcands;
   PrevFilterOutput->getObjects(TriggerCluster,  recoecalcands);
   if(recoecalcands.empty()) PrevFilterOutput->getObjects(TriggerPhoton,recoecalcands);  //we dont know if its type trigger cluster or trigger photon
 
   // Refs to the two Candidate objects used to calculate deltaPhi
   edm::Ref<reco::RecoEcalCandidateCollection> ref1, ref2;

   // look at all candidates,  check cuts
   int n(0);
   for(unsigned int i=0; i<recoecalcands.size(); i++) {
      const edm::Ref<reco::RecoEcalCandidateCollection> & ref = recoecalcands[i];
      if( ref->et() >= etcut_) {
	++n;
	if(n==1)  ref1 = ref;
	if(n==2)  ref2 = ref;
      }
   }

   // if there are only two Candidate objects, calculate deltaPhi  
   double deltaPhi(0.0);
   if(n==2) {
      deltaPhi = fabs(ref1->phi()-ref2->phi());
      if(deltaPhi>M_PI) deltaPhi = 2*M_PI - deltaPhi;

      filterproduct.addObject(TriggerCluster, ref1);
      filterproduct.addObject(TriggerCluster, ref2);  
   } 
        
   // filter decision
   bool accept(n==2 && deltaPhi>minDeltaPhi_);
  
   return accept;
}

// define as a framework module
// #include "FWCore/Framework/interface/MakerMacros.h"
// DEFINE_FWK_MODULE(HLTEgammaDoubleEtDeltaPhiFilter);

