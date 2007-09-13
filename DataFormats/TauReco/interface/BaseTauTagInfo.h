#ifndef DataFormats_TauReco_BaseTauTagInfo_h
#define DataFormats_TauReco_BaseTauTagInfo_h

/* class BaseTauTagInfo
 * base class 
 * author: Ludovic Houchu (Ludovic.Houchu@cern.ch)
 * created: Sep 4 2007,
 * revised: 
 */

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TauReco/interface/BaseTauTagInfoFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

using namespace std;
using namespace edm;
using namespace reco;

namespace reco{ 
  class BaseTauTagInfo {
  public:
    BaseTauTagInfo();
    virtual ~BaseTauTagInfo(){};
    
    // Tracks which are components of JetTracksAssociation object and which were filtered by RecoTauTag/TauTagTools/ TauTagTools::filteredTracks(.,...) function through RecoTauTag/RecoTauTag/ CaloRecoTauTagInfoProducer or PFRecoTauTagInfoProducer EDProducer
    const TrackRefVector& Tracks()const;
    void setTracks(const TrackRefVector);
    
    // rec. jet Lorentz-vector combining (Tracks and neutral ECAL Island BasicClusters) or (charged hadr. PFCandidates and gamma PFCandidates)
    const math::XYZTLorentzVector alternatLorentzVect()const;
    void setalternatLorentzVect(math::XYZTLorentzVector);
  protected:
    TrackRefVector Tracks_;
    math::XYZTLorentzVector alternatLorentzVect_;
  };
}

#endif

