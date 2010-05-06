#ifndef RecoParticleFlow_PFTracking_PFDisplacedVertexHelper_h
#define RecoParticleFlow_PFTracking_PFDisplacedVertexHelper_h 

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"

/// \brief Displaced Vertex Finder Algorithm
/*!
  \author Maxime Gouzevitch
  \date October 2009
*/

class PFDisplacedVertexHelper {

 public:

  PFDisplacedVertexHelper(); 
  ~PFDisplacedVertexHelper();

  /// Set Tracks selector parameters
  void setTracksSelector(const edm::ParameterSet& ps){
    tracksSelector_ = TracksSelector(ps);
  };

  /// Set Vertex identifier parameters
  void setVertexIdentifier(const edm::ParameterSet& ps){
    vertexIdentifier_ = VertexIdentifier(ps);
  };

  /// Update the primary vertex information
  void setPrimaryVertex(edm::Handle< reco::VertexCollection > mainVertexHandle, 
			edm::Handle< reco::BeamSpot > beamSpotHandle);


  /// Select tracks tool
  bool isTrackSelected(const reco::Track& trk, 
		       const reco::PFDisplacedVertex::VertexTrackType vertexTrackType) const;

  /// Vertex identification tool
  reco::PFDisplacedVertex::VertexType identifyVertex(const reco::PFDisplacedVertex& v) const;

  /// Set Vertex direction using the primary vertex
  math::XYZPoint primaryVertex() const { return pvtx_;}

  void Dump(std::ostream& out = std::cout) const;

 private:


  /// Tools used to calculate quantities for vertex identification
  double    angle(const reco::PFDisplacedVertex& v) const;
  int    lambdaCP(const reco::PFDisplacedVertex& v) const;
  bool isKaonMass(const reco::PFDisplacedVertex& v) const;


  /// Tool which store the information for the tracks selection
  struct TracksSelector {
    TracksSelector() : 
      bSelectTracks_(false),
      nChi2_min_(0), nChi2_max_(100), 
      pt_min_(0), dxy_min_(0), 
      nHits_min_(3), nOuterHits_max_(100),
      quality_("loose"){}
  
    TracksSelector(const edm::ParameterSet& ps){
      bSelectTracks_  = ps.getParameter<bool>("bSelectTracks");
      nChi2_min_      = ps.getParameter<double>("nChi2_min");
      nChi2_max_      = ps.getParameter<double>("nChi2_max");
      pt_min_         = ps.getParameter<double>("pt_min");
      dxy_min_        = ps.getParameter<double>("dxy_min");
      nHits_min_      = ps.getParameter<int>("nHits_min");
      nOuterHits_max_ = ps.getParameter<int>("nOuterHits_max");
      std::string quality_ = ps.getParameter<std::string>("quality");
    }
    
    bool selectTracks() const {return bSelectTracks_;}
    double nChi2_min() const {return nChi2_min_;}
    double nChi2_max() const {return nChi2_max_;}
    double pt_min() const {return pt_min_;}
    double dxy_min() const {return dxy_min_;}
    int nHits_min() const {return nHits_min_;}
    int nOuterHits_max() const {return nOuterHits_max_;}
    std::string quality() const {return quality_;}
    double dxy(const reco::Track& trk) const {return trk.dxy(pvtx_);}

    bool bSelectTracks_;
    double nChi2_min_;
    double nChi2_max_;
    double pt_min_;
    double dxy_min_;
    int nHits_min_;
    int nOuterHits_max_;
    math::XYZPoint pvtx_;
    std::string quality_;

    void Dump(std::ostream& out = std::cout) const {
      if(! out ) return;
      std::string s =  bSelectTracks_ ? "On" : "Off";

      out << "" << std::endl;
      out << "      ==== The TrackerSelector is " << s.data() << " ====    " << std::endl;

      out << " nChi2_min_ = " << nChi2_min_
	  << " nChi2_max_ = " << nChi2_max_ << std::endl
	  << " pt_min_ = " << pt_min_
	  << " dxy_min_ = " << dxy_min_ << std::endl
	  << " nHits_min_ = " << nHits_min_  
	  << " nOuterHits_max_ = " << nOuterHits_max_ << std::endl
	  << " quality = " << quality_ << std::endl; 
    
    }

  };

  /// Tool which store the information for the vertex identification
  struct VertexIdentifier {
    VertexIdentifier():
      bIdentifyVertices_(false),
      pt_min_(0.2),
      pt_kink_min_(1.4),
      looper_eta_max_(0.1),
      logPrimSec_min_(0.2){

      double m[] = {0.050, 0.470, 0.525, 0.470, 0.525, 1.107, 1.125, 0.200};
      std::vector< double > masses(m, m+8);
      masses_ = masses;

      double a[] = {60, 40};
      std::vector< double > angles(a, a+1);
      angles_ = angles;

    };
  
    VertexIdentifier(const edm::ParameterSet& ps){
      bIdentifyVertices_  = ps.getParameter<bool>("bIdentifyVertices");
      angles_             = ps.getParameter< std::vector<double> >("angles");
      masses_             = ps.getParameter< std::vector<double> >("masses");
      pt_min_             = ps.getParameter<double>("pt_min");
      pt_kink_min_        = ps.getParameter<double>("pt_kink_min");
      looper_eta_max_     = ps.getParameter<double>("looper_eta_max");
      logPrimSec_min_     = ps.getParameter<double>("logPrimSec_min");
    }

    bool identifyVertices() const {return bIdentifyVertices_;}

    double angle_max() const {return angles_[0];}
    double angle_V0Conv_max() const {return angles_[1];}
    
    double pt_min() const {return pt_min_;}
    double pt_kink_min() const {return pt_kink_min_;}

    double mConv_max() const {return masses_[0];}
    double mK0_min() const {return masses_[1];}
    double mK0_max() const {return masses_[2];}
    double mK_min() const {return masses_[3];}
    double mK_max() const {return masses_[4];}
    double mLambda_min() const {return masses_[5];}
    double mLambda_max() const {return masses_[6];}
    double mNucl_min() const {return masses_[7];}
    
    double looper_eta_max() const {return looper_eta_max_;}
    double logPrimSec_min() const {return logPrimSec_min_;}

    bool bIdentifyVertices_;
    std::vector<double> angles_;
    std::vector<double> masses_;
    double pt_min_;
    double pt_kink_min_;
    double looper_eta_max_;
    double logPrimSec_min_;

    void Dump(std::ostream& out = std::cout) const {
      if(! out ) return;
      std::string s =  bIdentifyVertices_ ? "On" : "Off";
      out << "" << std::endl;
      out << "      ==== The Vertex Identifier is " << s.data() << " ====    " << std::endl;

      out << " pt_min_ = " << pt_min_
	  << " pt_kink_min_ = " << pt_kink_min_ << std::endl
	  << " looper_eta_max_ = " << looper_eta_max_  
	  << " log10(P_Prim/P_Sec)_min " << logPrimSec_min_ << std::endl   
	  << " Mass_conv > " << mConv_max() << std::endl
	  << " " << mK0_min() << " < Mass_K0 < " << mK0_max() << std::endl
	  << " " << mK_min() << " < Mass_K+- < " << mK_max() << std::endl
	  << " " << mLambda_min() << " < Mass_Lambda < " << mLambda_max() << std::endl
	  << " Mass_Nucl_ee > " << mNucl_min() << std::endl
	  << " angle_max = " << angle_max() 
	  << " angle_V0Conv_max = " << angle_V0Conv_max()  << std::endl; 
    
    }



  };


  TracksSelector tracksSelector_;
  VertexIdentifier vertexIdentifier_;
  /// Primary vertex information updated for each event
  math::XYZPoint pvtx_;

  /// Masses2 taken from PDG
  static const double pion_mass2;
  static const double muon_mass2;
  static const double proton_mass2;

};

#endif


