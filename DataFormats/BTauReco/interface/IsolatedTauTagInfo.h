#ifndef BTauReco_TauTagIsolation_h
#define BTauReco_TauTagIsolation_h
//
// \class IsolatedTauTagInfo
// \short Extended object for the Tau Isolation algorithm.
// contains the result and the methods used in the ConeIsolation Algorithm, to create the 
// object to be made persistent on RECO
//
// \author: Simone Gennai, based on ORCA class by S. Gennai and F. Moortgat
//


#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfoFwd.h"
//#include "DataFormats/TrackReco/interface/HelixParameters.h"
#include "DataFormats/Math/interface/Vector3D.h"
//Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
//
using namespace std;

namespace reco { 


class IsolatedTauTagInfo : public JTATagInfo {

  public:
    //default constructor
    IsolatedTauTagInfo() {}


    IsolatedTauTagInfo(edm::RefVector<TrackCollection> tracks,const JetTracksAssociationRef & jtaRef):JTATagInfo(jtaRef) 
      {    
	track_iterator it = tracks.begin();
	for(;it!= tracks.end(); it++)
	  {
	    selectedTracks_.push_back(*it);
	  }
	
    }
    //destructor
    virtual ~IsolatedTauTagInfo() {};
    
    //get the tracks from the jetTag
    const TrackRefVector allTracks() const { return tracks(); }

    //get the selected tracks used to computed the isolation
    const TrackRefVector & selectedTracks() const {return selectedTracks_;}
    
    virtual IsolatedTauTagInfo* clone() const { return new IsolatedTauTagInfo( *this ); }
  
    //default discriminator: returns the value of the discriminator of the jet tag, i.e. the one computed with the parameters taken from the cfg file
    //   using JTATagInfo::discriminator;
    float  discriminator() const {return -1.; }
     
    //methods to be used to recomputed the isolation with a new set of parameters
    float discriminator(float m_cone, float sig_cone, float iso_con, float pt_min_lt, float pt_min_tk, int nTracksIsoRing = 0) const;
    float discriminator( math::XYZVector myVector, float m_cone, float sig_cone, float iso_con, float pt_min_lt, float pt_min_tk, int nTracksIsoRing) const;
    //Used in case the PV is not considered
    float discriminator(float m_cone, float sig_cone, float iso_con, float pt_min_lt, float pt_min_tk, int nTracksIsoRing, float dz_lt) const;
    float discriminator( math::XYZVector myVector, float m_cone, float sig_cone, float iso_con, float pt_min_lt, float pt_min_tk, int nTracksIsoRing, float dz_lt) const;
    
    // return all tracks in a cone of size "size" around a direction "direction" 
    const edm::RefVector<TrackCollection> tracksInCone(const math::XYZVector myVector,const float size,  const float pt_min ) const;
    const edm::RefVector<TrackCollection> tracksInCone(const math::XYZVector myVector,const float size,  const float pt_min, const float z_pv, const float dz_lt ) const;
    
    //return the leading track in a given cone around the jet axis or a given direction
    const TrackRef leadingSignalTrack(const float rm_cone, const float pt_min) const;
    const TrackRef leadingSignalTrack(math::XYZVector myVector, const float rm_cone, const float pt_min) const;
     
  private:
    TrackRefVector selectedTracks_;
  };
}

#endif
