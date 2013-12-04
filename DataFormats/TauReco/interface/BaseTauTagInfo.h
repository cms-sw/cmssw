#ifndef DataFormats_TauReco_BaseTauTagInfo_h
#define DataFormats_TauReco_BaseTauTagInfo_h

/* class BaseTauTagInfo
 * base class 
 * author: Ludovic Houchu (Ludovic.Houchu@cern.ch)
 * created: Sep 4 2007,
 * revised: 
 */

#include "DataFormats/TauReco/interface/BaseTauTagInfoFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/JetReco/interface/CaloJetCollection.h"


namespace reco{ 
  class BaseTauTagInfo {
  public:
    BaseTauTagInfo();
    virtual ~BaseTauTagInfo(){};
    
    // Tracks which are components of JetTracksAssociation object and which were filtered by RecoTauTag/TauTagTools/ TauTagTools::filteredTracks(.,...) function through RecoTauTag/RecoTauTag/ CaloRecoTauTagInfoProducer or PFRecoTauTagInfoProducer EDProducer
    const reco::TrackRefVector& Tracks()const;
    void setTracks(const TrackRefVector&);
  protected:
    reco::TrackRefVector Tracks_;
  };
}

#endif

