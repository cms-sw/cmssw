#ifndef RecoTauTag_TauTagTools_TauTagTools_h
#define RecoTauTag_TauTagTools_TauTagTools_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "CommonTools/Utils/interface/Angle.h"

#include "Math/GenVector/VectorUtil.h" 

#include "RecoTauTag/TauTagTools/interface/ECALBounds.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include "TFormula.h"

namespace TauTagTools{
  template <class T> class sortByOpeningDistance;
  reco::TrackRefVector filteredTracksByNumTrkHits(reco::TrackRefVector theInitialTracks, int tkminTrackerHitsn);
  reco::TrackRefVector filteredTracks(reco::TrackRefVector theInitialTracks,double tkminPt,int tkminPixelHitsn,int tkminTrackerHitsn,double tkmaxipt,double tkmaxChi2, reco::Vertex pV);
  reco::TrackRefVector filteredTracks(reco::TrackRefVector theInitialTracks,double tkminPt,int tkminPixelHitsn,int tkminTrackerHitsn,double tkmaxipt,double tkmaxChi2,double tktorefpointmaxDZ,reco::Vertex pV,double refpoint_Z);

  std::vector<reco::PFCandidatePtr> filteredPFChargedHadrCandsByNumTrkHits(std::vector<reco::PFCandidatePtr> theInitialPFCands, int ChargedHadrCand_tkminTrackerHitsn);
  std::vector<reco::PFCandidatePtr> filteredPFChargedHadrCands(std::vector<reco::PFCandidatePtr> theInitialPFCands,double ChargedHadrCand_tkminPt,int ChargedHadrCand_tkminPixelHitsn,int ChargedHadrCand_tkminTrackerHitsn,double ChargedHadrCand_tkmaxipt,double ChargedHadrCand_tkmaxChi2, reco::Vertex pV);
  std::vector<reco::PFCandidatePtr> filteredPFChargedHadrCands(std::vector<reco::PFCandidatePtr> theInitialPFCands,double ChargedHadrCand_tkminPt,int ChargedHadrCand_tkminPixelHitsn,int ChargedHadrCand_tkminTrackerHitsn,double ChargedHadrCand_tkmaxipt,double ChargedHadrCand_tkmaxChi2,double ChargedHadrCand_tktorefpointmaxDZ,reco::Vertex pV, double refpoint_Z);
  std::vector<reco::PFCandidatePtr> filteredPFNeutrHadrCands(std::vector<reco::PFCandidatePtr> theInitialPFCands,double NeutrHadrCand_HcalclusMinEt);
  std::vector<reco::PFCandidatePtr> filteredPFGammaCands(std::vector<reco::PFCandidatePtr> theInitialPFCands,double GammaCand_EcalclusMinEt);
  math::XYZPoint propagTrackECALSurfContactPoint(const MagneticField*,reco::TrackRef); 

  TFormula computeConeSizeTFormula(const std::string& ConeSizeFormula,const char* errorMessage);
  void replaceSubStr(std::string& s,const std::string& oldSubStr,const std::string& newSubStr); 

  double computeDeltaR(const math::XYZVector& vec1, const math::XYZVector& vec2);
  double computeAngle(const math::XYZVector& vec1, const math::XYZVector& vec2);

  //MIKE: Sort a reference vector
  void sortRefVectorByPt(std::vector<reco::PFCandidatePtr>&);


  //binary predicate classes for sorting a list of indexes corresponding to PFPtrVectors...(as they can't use STL??)
  class sortRefsByOpeningDistance
  {
     public:
     sortRefsByOpeningDistance(const math::XYZVector& theAxis, double (*ptrToMetricFunction)(const math::XYZVector&, const math::XYZVector&), const std::vector<reco::PFCandidatePtr>& myInputVector):myMetricFunction(ptrToMetricFunction),axis(theAxis),myVector(myInputVector){};
     bool operator()(uint32_t indexA, uint32_t indexB)
     {
        const reco::PFCandidatePtr candA = myVector.at(indexA);
        const reco::PFCandidatePtr candB = myVector.at(indexB);
        return (myMetricFunction(axis, candA->momentum()) < myMetricFunction(axis, candB->momentum()));
     }
     private:
     double (*myMetricFunction)(const math::XYZVector&, const math::XYZVector&);
     math::XYZVector axis;  //axis about which candidates are sorted
     const std::vector<reco::PFCandidatePtr> myVector;
  };
  class filterChargedAndNeutralsByPt
  {
     public:
     filterChargedAndNeutralsByPt(double minNeutralPt, double minChargedPt, const std::vector<reco::PFCandidatePtr>& myInputVector):minNeutralPt_(minNeutralPt),minChargedPt_(minChargedPt),myVector(myInputVector){};
     bool operator()(uint32_t candIndex)
     {
        const reco::PFCandidatePtr cand = myVector.at(candIndex);
        bool output          = true;
        unsigned char charge = std::abs(cand->charge());
        double thePt         = cand->pt();
        if (charge && thePt < minChargedPt_)
           output = false;
        else if (!charge && thePt < minNeutralPt_)
           output = false;
        return output;
     }
     private:
     double minNeutralPt_;
     double minChargedPt_;
     const std::vector<reco::PFCandidatePtr>& myVector;
  };

  class refVectorPtSorter {
  public:
    refVectorPtSorter(const std::vector<reco::PFCandidatePtr>& vec)
      {
	vec_ = vec;
      }

    refVectorPtSorter()
      {
      }


    ~refVectorPtSorter()
      {}

    bool operator()(size_t a , size_t b) {
      return (vec_.at(a)->pt() > vec_.at(b)->pt());
    }

  private:
    std::vector<reco::PFCandidatePtr> vec_;
  };


  //binary predicate classes for sorting vectors of candidates
  template <class T>
  class sortByAscendingPt
  {
     public:
     bool operator()(const T& candA, const T& candB)
     {
        return (candA.pt() > candB.pt());
     }
     bool operator()(const T* candA, const T* candB)
     {
        return (candA->pt() > candB->pt());
     }
  };

  template <class T>
  class sortByDescendingPt
  {
     public:
     bool operator()(const T& candA, const T& candB)
     {
        return (candA.pt() < candB.pt());
     }
     bool operator()(const T* candA, const T* candB)
     {
        return (candA->pt() < candB->pt());
     }
  };

  template <class T>
  class sortByOpeningAngleAscending
  {
     public:
     sortByOpeningAngleAscending(const math::XYZVector& theAxis, double (*ptrToMetricFunction)(const math::XYZVector&, const math::XYZVector&)):axis(theAxis),myMetricFunction(ptrToMetricFunction){};
     bool operator()(const T& candA, const T& candB)
     {
        return ( myMetricFunction(axis, candA.momentum()) > myMetricFunction(axis, candB.momentum()) );
     }
     bool operator()(const T* candA, const T* candB)
     {
        return ( myMetricFunction(axis, candA->momentum()) > myMetricFunction(axis, candB->momentum()) );
     }
     private:
        math::XYZVector axis;
        double (*myMetricFunction)(const math::XYZVector&, const math::XYZVector&);
  };

  template <class T>
  class sortByOpeningAngleDescending
  {
     public:
     sortByOpeningAngleDescending(const math::XYZVector& theAxis, double (*ptrToMetricFunction)(const math::XYZVector&, const math::XYZVector&)):axis(theAxis),myMetricFunction(ptrToMetricFunction){};
     bool operator()(const T& candA, const T& candB)
     {
        return ( myMetricFunction(axis, candA.momentum()) < myMetricFunction(axis, candB.momentum()) );
     }
     bool operator()(const T* candA, const T* candB)
     {
        return ( myMetricFunction(axis, candA->momentum()) < myMetricFunction(axis, candB->momentum()) );
     }
     private:
        math::XYZVector axis;
        double (*myMetricFunction)(const math::XYZVector&, const math::XYZVector&);
  };

}

#endif

