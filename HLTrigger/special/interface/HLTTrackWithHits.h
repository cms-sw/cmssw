#ifndef HLTrigger_HLTTrackWithHits_H
/**\class HLTTrackWithHits
 * Description:
 * templated EDFilter to count the number of tracks with a given hit requirement
 * \author Jean-Roch Vlimant
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageService/interface/MessageLogger.h"

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class HLTTrackWithHits : public HLTFilter {
public:
  explicit HLTTrackWithHits(const edm::ParameterSet& iConfig) :
    src_(iConfig.getParameter<edm::InputTag>("src")),
    minN_(iConfig.getParameter<int>("MinN")),
    maxN_(iConfig.getParameter<int>("MaxN")),
    MinBPX_(iConfig.getParameter<int>("MinBPX")),
    MinFPX_(iConfig.getParameter<int>("MinFPX")),
    MinPXL_(iConfig.getParameter<int>("MinPXL"))
      {
	produces<trigger::TriggerFilterObjectWithRefs>();
      };
  
  ~HLTTrackWithHits(){};
  
private:
  virtual bool filter(edm::Event& iEvent, const edm::EventSetup&)
  {
    // The filtered object. which is put empty.
    std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));

    edm::Handle<reco::TrackCollection> oHandle;
    iEvent.getByLabel(src_, oHandle);
    int s=oHandle->size();
    int count=0;
    for (int i=0;i!=s;++i){
      const reco::Track & track = (*oHandle)[i];
      const reco::HitPattern & hits = track.hitPattern();
      if ( MinBPX_>0 && hits.numberOfValidPixelBarrelHits() >= MinBPX_ ) count++; continue;
      if ( MinFPX_>0 && hits.numberOfValidPixelEndcapHits() >= MinFPX_ ) count++; continue;
      if ( MinPXL_>0 && hits.numberOfValidPixelHits() >= MinPXL_ ) count++; continue;
    }
      
    bool answer=(count>=minN_ && count<=maxN_);
    LogDebug("HLTTrackWithHits")<<module()<<" sees: "<<s<<" objects. Only: "<<count<<" satisfy the hit requirement. Filter answer is: "<<(answer?"true":"false")<<std::endl;

    iEvent.put(filterproduct);
    return answer;
  }
  virtual void endJob(){};
 
  edm::InputTag src_;
  int minN_,maxN_,MinBPX_,MinFPX_,MinPXL_;
};


#endif
