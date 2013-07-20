#ifndef EgammaCandidates_Conversion_h
#define EgammaCandidates_Conversion_h
/** \class reco::Conversion Conversion.h DataFormats/EgammaCandidates/interface/Conversion.h
 *
 * 
 *
 * \author N.Marinelli  University of Notre Dame, US
 *
 * \version $Id: Conversion.h,v 1.25 2013/04/22 22:44:45 wmtan Exp $
 *
 */

#include <bitset>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1DFloat.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"


namespace reco {
    class Conversion  {
  public:

      enum ConversionAlgorithm {undefined=0, 
				ecalSeeded=1, 
				trackerOnly=2, 
				mixed=3, 
                                pflow=4,
				algoSize=5}; 

      enum ConversionQuality {generalTracksOnly=0, 
			      arbitratedEcalSeeded=1, 
			      arbitratedMerged=2,
			      arbitratedMergedEcalGeneral=3,
			      highPurity=8, 
			      highEfficiency=9,
			      ecalMatched1Track=10,
			      ecalMatched2Track=11};

      static const std::string algoNames[];      

      // Default constructor
      Conversion();
      
      Conversion( const reco::CaloClusterPtrVector clu, 
		  const std::vector<edm::RefToBase<reco::Track> >& tr,
		  const std::vector<math::XYZPointF>& trackPositionAtEcal , 
		  const reco::Vertex  &  convVtx,
		  const std::vector<reco::CaloClusterPtr> & matchingBC,
		  const float DCA,        
		  const std::vector<math::XYZPointF> & innPoint,
		  const std::vector<math::XYZVectorF> & trackPin ,
		  const std::vector<math::XYZVectorF> & trackPout,
                  const std::vector<uint8_t>& nHitsBeforeVtx,
                  const std::vector<Measurement1DFloat> & dlClosestHitToVtx,
                  uint8_t nSharedHits,
                  const float mva,
		  ConversionAlgorithm=undefined);


      Conversion( const reco::CaloClusterPtrVector clu, 
		  const std::vector<reco::TrackRef>& tr,
		  const std::vector<math::XYZPointF>& trackPositionAtEcal , 
		  const reco::Vertex  &  convVtx,
		  const std::vector<reco::CaloClusterPtr> & matchingBC,
		  const float DCA,        
		  const std::vector<math::XYZPointF> & innPoint,
		  const std::vector<math::XYZVectorF> & trackPin ,
		  const std::vector<math::XYZVectorF> & trackPout,
                  const float mva,
		  ConversionAlgorithm=undefined);
      


      
      Conversion( const reco::CaloClusterPtrVector clu, 
		  const std::vector<reco::TrackRef>& tr,
		  const reco::Vertex  &  convVtx,
		  ConversionAlgorithm=undefined);
      
      Conversion( const reco::CaloClusterPtrVector clu, 
		  const std::vector<edm::RefToBase<reco::Track> >& tr,
		  const reco::Vertex  &  convVtx,
		  ConversionAlgorithm=undefined);
      
           
      
      /// destructor
      virtual ~Conversion();
      /// returns a clone of the candidate
      Conversion * clone() const;
      /// Pointer to CaloCluster (foe Egamma Conversions it points to  a SuperCluster)
      reco::CaloClusterPtrVector caloCluster() const {return caloCluster_ ;}
      /// vector of track to base references 
      std::vector<edm::RefToBase<reco::Track> > tracks() const ; 
      /// returns  the reco conversion vertex
      const reco::Vertex & conversionVertex() const  { return theConversionVertex_ ; }
      /// Bool flagging objects having track size >0
      bool isConverted() const;
      /// Number of tracks= 0,1,2
      unsigned int nTracks() const {return  tracks().size(); }
      /// get the value  of the TMVA output
      double MVAout() const { return theMVAout_;}
      /// get the MVS output from PF for one leg conversions
      std::vector<float> const oneLegMVA() {return theOneLegMVA_;}
      /// if nTracks=2 returns the pair invariant mass. Original tracks are used here
      double pairInvariantMass() const;
      /// Delta cot(Theta) where Theta is the angle in the (y,z) plane between the two tracks. Original tracks are used
      double pairCotThetaSeparation() const;
      /// Conversion tracks momentum from the tracks inner momentum
      math::XYZVectorF  pairMomentum() const;
      /// Conversion track pair 4-momentum from the tracks refitted with vertex constraint
      math::XYZTLorentzVectorF   refittedPair4Momentum() const;
      /// Conversion tracks momentum from the tracks refitted with vertex constraint
      math::XYZVectorF  refittedPairMomentum() const;
      /// Super Cluster energy divided by track pair momentum if Standard seeding method. If a pointer to two (or more clusters)
      /// is stored in the conversion, this method returns the energy sum of clusters divided by the track pair momentum
      /// Track innermost momentum is used here
      double EoverP() const;
      /// Super Cluster energy divided by track pair momentum if Standard seeing method. If a pointer to two (or more clusters)
      /// is stored in the conversion, this method returns the energy sum of clusters divided by the track pair momentum
      ///  Track momentum refitted with vertex constraint is used
      double EoverPrefittedTracks() const;
      // Dist of minimum approach between tracks
      double distOfMinimumApproach() const {return  theMinDistOfApproach_;}
      // deltaPhi tracks at innermost point
      double dPhiTracksAtVtx() const;
      // deltaPhi tracks at ECAl
      double dPhiTracksAtEcal() const;
      // deltaEta tracks at ECAl
      double dEtaTracksAtEcal() const;

      //impact parameter and decay length computed with respect to given beamspot or vertex
      //computed from refittedPairMomentum

      //transverse impact parameter
      double dxy(const math::XYZPoint& myBeamSpot = math::XYZPoint()) const;
      //longitudinal impact parameter
      double dz(const math::XYZPoint& myBeamSpot = math::XYZPoint()) const;
      //transverse decay length
      double lxy(const math::XYZPoint& myBeamSpot = math::XYZPoint()) const;
      //longitudinal decay length
      double lz(const math::XYZPoint& myBeamSpot = math::XYZPoint()) const;
      //z position of intersection with beamspot in rz plane (possible tilt of beamspot is neglected)
      double zOfPrimaryVertexFromTracks(const math::XYZPoint& myBeamSpot = math::XYZPoint()) const { return dz(myBeamSpot) + myBeamSpot.z(); }

      ///// The following are variables provided per each track
      /// positions of the track extrapolation at the ECAL front face
      const std::vector<math::XYZPointF> & ecalImpactPosition() const  {return thePositionAtEcal_;}
      //  pair of BC matching a posteriori the tracks
      const std::vector<reco::CaloClusterPtr>&  bcMatchingWithTracks() const { return theMatchingBCs_;}
      /// signed transverse impact parameter for each track
      std::vector<double> tracksSigned_d0() const ;
      /// Vector containing the position of the innermost hit of each track
      const std::vector<math::XYZPointF>& tracksInnerPosition() const {return theTrackInnerPosition_;}
      /// Vector of track momentum measured at the outermost hit
      const std::vector<math::XYZVectorF>& tracksPout() const {return theTrackPout_;}
      /// Vector of track momentum measured at the innermost hit
      const std::vector<math::XYZVectorF>& tracksPin() const  {return theTrackPin_;}
      ///Vector of the number of hits before the vertex along each track trajector
      const std::vector<uint8_t> &nHitsBeforeVtx() const { return nHitsBeforeVtx_; }
      ///Vector of signed decay length with uncertainty from nearest hit on track to the conversion vtx positions
      const std::vector<Measurement1DFloat> &dlClosestHitToVtx() const { return dlClosestHitToVtx_; }
      ///number of shared hits btw the two track
      uint8_t nSharedHits() const { return nSharedHits_; }


      /// set the value  of the TMVA output
      void setMVAout(const float& mva) { theMVAout_=mva;}
      /// set the MVS output from PF for one leg conversions
      void setOneLegMVA(const std::vector<float>& mva) { theOneLegMVA_=mva;}
      // Set the ptr to the Super cluster if not set in the constructor 
      void setMatchingSuperCluster ( const  reco::CaloClusterPtrVector& sc) { caloCluster_= sc;}
      /// Conversion Track algorithm/provenance
      void setConversionAlgorithm(const ConversionAlgorithm a, bool set=true) { if (set) algorithm_=a; else algorithm_=undefined;}
      ConversionAlgorithm algo() const ;
      std::string algoName() const;
      static std::string algoName(ConversionAlgorithm );
      static ConversionAlgorithm  algoByName(const std::string &name);      

      bool quality(ConversionQuality q) const { return  (qualityMask_ & (1<<q))>>q; }
      void setQuality(ConversionQuality q, bool b);


      
    private:
      
      /// vector pointer to a/multiple seed CaloCluster(s)
      reco::CaloClusterPtrVector caloCluster_;
      ///  vector of Track references
      std::vector<reco::TrackRef>  tracks_;
      /// vector Track RefToBase
      mutable std::vector<edm::RefToBase<reco::Track> >  trackToBaseRefs_;
      /// position at the ECAl surface of the track extrapolation
      std::vector<math::XYZPointF>  thePositionAtEcal_;
      /// Fitted Kalman conversion vertex
      reco::Vertex theConversionVertex_;
      /// Clusters mathing the tracks (these are not the seeds)
      std::vector<reco::CaloClusterPtr> theMatchingBCs_;
      /// Distance of min approach of the two tracks
      float theMinDistOfApproach_;
      /// P_in of tracks
      std::vector<math::XYZPointF> theTrackInnerPosition_;    
      /// P_in of tracks
      std::vector<math::XYZVectorF> theTrackPin_;    
      /// P_out of tracks
      std::vector<math::XYZVectorF> theTrackPout_;    
      ///number of hits before the vertex on each trackerOnly
      std::vector<uint8_t> nHitsBeforeVtx_;
      ///signed decay length and uncertainty from nearest hit on track to conversion vertex
      std::vector<Measurement1DFloat> dlClosestHitToVtx_;
      ///number of shared hits between tracks
      uint8_t nSharedHits_;
      /// TMVA output
      float theMVAout_;
      /// vectors of TMVA outputs from pflow for one leg conversions
      std::vector<float>  theOneLegMVA_;
      /// conversion algorithm/provenance
      uint8_t algorithm_;
      uint16_t qualityMask_;


  };

    inline Conversion::ConversionAlgorithm Conversion::algo() const {
      return (ConversionAlgorithm) algorithm_;
    }
    
    
    inline std::string Conversion::algoName() const{
            
      switch(algorithm_)
	{
	case undefined: return "undefined";
	case ecalSeeded: return "ecalSeeded";
	case trackerOnly: return "trackerOnly";
	case mixed: return "mixed";
	case pflow: return "pflow";

	}
      return "undefined";
    }

    inline std::string Conversion::algoName(ConversionAlgorithm a){
      if(int(a) < int(algoSize) && int(a)>0) return algoNames[int(a)];
      return "undefined";
    }

    inline void Conversion::setQuality(ConversionQuality q, bool b){
      if (b)//regular OR if setting value to true
        qualityMask_ |= (1<<q) ;
      else // doing "half-XOR" if unsetting value
        qualityMask_ &= (~(1<<q));

    }
  
}

#endif
