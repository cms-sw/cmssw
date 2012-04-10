#ifndef ggPFTracks_h
#define ggPFTracks_h
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "TVector3.h"
using namespace edm;
using namespace std;
using namespace reco;

class ggPFTracks{
 public:
  explicit ggPFTracks(
		      edm::Handle<BeamSpot>& beamSpotHandle
		      );
  virtual ~ggPFTracks();
  
  virtual void getPFConvTracks(
			       reco::Photon phot,
			       //reco::PFCandidate PFCand, 
			       vector<edm::RefToBase<reco::Track> > &Tracks, 
			       reco::ConversionRefVector &conversions,
			       vector<edm::RefToBase<reco::Track> > &SLTracks, 
			       reco::ConversionRefVector &SLconversions
			       );
  std::pair<float,float> BeamLineInt(
		    reco::SuperClusterRef sc,
		    vector<edm::RefToBase<reco::Track> > &Tracks, 
		    reco::ConversionRefVector &conversions,
		    vector<edm::RefToBase<reco::Track> > &SLTracks, 
		    reco::ConversionRefVector &SLconversions
		    );
  std::pair<float,float> gsfTrackProj(
				      reco::GsfTrackRef gsf
				      );
  std::pair<float,float> gsfElectronProj(
					 reco::GsfElectron gsf
					 );
  std::pair<float,float> TrackProj(
				   bool isEb,
				   reco::GsfTrackRef gsf,
				   vector<edm::RefToBase<reco::Track> > &SLTracks, 
				   reco::ConversionRefVector &SLconversions
		  );
  std::pair<float, float> CombZVtx(
		 reco::SuperClusterRef sc, 
		 reco::GsfTrackRef gsf,
		 vector<edm::RefToBase<reco::Track> > &Tracks, 
		 reco::ConversionRefVector &conversions,
		 vector<edm::RefToBase<reco::Track> > &SLTracks, 
		 reco::ConversionRefVector &SLconversions
		 );
  std::pair<float, float> SLCombZVtx(
				     reco::Photon phot,
				     bool &hasSL
				     );
  bool isConv(){return isConv_;}
  

 private:

  Handle<BeamSpot> beamSpotHandle_;
  bool isConv_;  
};
#endif
