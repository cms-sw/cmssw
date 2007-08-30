#ifndef DataFormats_TauReco_TauTagInfo_h
#define DataFormats_TauReco_TauTagInfo_h

/* class TauTagInfo
 * the object of this class is directly created by RecoTauTag/RecoTau CaloRecoTauTagInfoProducer EDProducer starting from JetTrackAssociations <a CaloJet,a list of Track's> object,
 *                          is the initial object for building a Tau object;
 * the object of the PFTauTagInfo class -whose TauTagInfo is the base class- is created by RecoTauTag/RecoTau PFRecoTauTagInfoProducer EDProducer starting from JetTrackAssociations <a PFJet,a list of Track's> object,
 *                          is the initial object for building a PFTau object;
 * created: Aug 29 2007,
 * revised: 
 * authors: Ludovic Houchu
 */

#include <math.h>

#include "Math/GenVector/PxPyPzE4D.h"

#include "DataFormats/TauReco/interface/TauTagInfoFwd.h"
#include "DataFormats/TrackReco/interface/Track.h" 
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

using namespace std;
using namespace edm;
using namespace reco;

namespace reco{ 
  class TauTagInfo{
  public:
    TauTagInfo(){
      alternatLorentzVect_.SetPx(NAN);
      alternatLorentzVect_.SetPy(NAN);
      alternatLorentzVect_.SetPz(NAN);
      alternatLorentzVect_.SetE(NAN);
    }
    virtual ~TauTagInfo(){};
    virtual TauTagInfo* clone()const{return new TauTagInfo(*this);}
    
    //get the rec. tk's which are components of JetTracksAssociator object which were filtered by RecoTauTag/RecoTau/ PFRecoTauTagInfoAlgorithm::filteredTracks or CaloRecoTauTagInfoAlgorithm::filteredTracks
    const TrackRefVector& Tracks()const{return Tracks_;}
    void  setTracks(const TrackRefVector x){Tracks_=x;}
    
    // rec. jet Lorentz-vector combining (Tracks and neutral ECAL BasicClusters) or (charged hadr. PFCandidate's and gamma PFCandidate's)
    math::XYZTLorentzVector alternatLorentzVect()const{return(alternatLorentzVect_);} 
    void setalternatLorentzVect(math::XYZTLorentzVector x){alternatLorentzVect_=x;}
    
    //the reference to the CaloJet
    const CaloJetRef& calojetRef()const{return CaloJetRef_;}
    void setcalojetRef(const CaloJetRef x){CaloJetRef_=x;}
  private:
    CaloJetRef CaloJetRef_;
    TrackRefVector Tracks_;
    math::XYZTLorentzVector alternatLorentzVect_;
  };
}

#endif

