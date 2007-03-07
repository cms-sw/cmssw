/** \class HLTElectronEoverpFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 * $Id: HLTElectronEoverpFilter.cc,v 1.3 2007/03/07 10:44:05 monicava Exp $
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronEoverpFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

//
// constructors and destructor
//
HLTElectronEoverpFilter::HLTElectronEoverpFilter(const edm::ParameterSet& iConfig)
{
   candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
   eoverpbarrelcut_  = iConfig.getParameter<double> ("eoverpbarrelcut");
   eoverpendcapcut_  = iConfig.getParameter<double> ("eoverpendcapcut");
   ncandcut_  = iConfig.getParameter<int> ("ncandcut");

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTElectronEoverpFilter::~HLTElectronEoverpFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTElectronEoverpFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // The filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  edm::RefToBase<reco::Candidate> candref; 


  // get hold of filtered candidates
  edm::Handle<reco::HLTFilterObjectWithRefs> electroncands;
  iEvent.getByLabel (candTag_,electroncands);
  
  // look at all candidates,  check cuts and add to filter object
  int n(0);

  
  for (unsigned int i=0; i<electroncands->size(); i++) {
    
    candref = electroncands->getParticleRef(i);
    
    reco::ElectronRef eleref = candref.castTo<reco::ElectronRef>();
    
    float elecEoverp = 0;
    const math::XYZVector trackMom =  eleref->track()->momentum();
    if( trackMom.R() != 0) elecEoverp = 
      eleref->superCluster()->energy()/ trackMom.R();

    if( fabs(eleref->eta()) < 1.5 ){
       if ( elecEoverp < eoverpbarrelcut_) {
	 n++;
	 filterproduct->putParticle(candref);
       }
     }
    if( (fabs(eleref->eta()) > 1.5) &&  (fabs(eleref->eta()) < 2.5) ){
       if ( elecEoverp < eoverpendcapcut_) {
	 n++;
	 filterproduct->putParticle(candref);
       }
     }

  }
  
  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}
