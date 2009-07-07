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


using namespace std;
using namespace reco;
using namespace edm;

namespace TauTagTools{
  template <class T> class sortByOpeningDistance;
  TrackRefVector filteredTracksByNumTrkHits(TrackRefVector theInitialTracks, int tkminTrackerHitsn);
  TrackRefVector filteredTracks(TrackRefVector theInitialTracks,double tkminPt,int tkminPixelHitsn,int tkminTrackerHitsn,double tkmaxipt,double tkmaxChi2, Vertex pV);
  TrackRefVector filteredTracks(TrackRefVector theInitialTracks,double tkminPt,int tkminPixelHitsn,int tkminTrackerHitsn,double tkmaxipt,double tkmaxChi2,double tktorefpointmaxDZ,Vertex pV,double refpoint_Z);

  PFCandidateRefVector filteredPFChargedHadrCandsByNumTrkHits(PFCandidateRefVector theInitialPFCands, int ChargedHadrCand_tkminTrackerHitsn);
  PFCandidateRefVector filteredPFChargedHadrCands(PFCandidateRefVector theInitialPFCands,double ChargedHadrCand_tkminPt,int ChargedHadrCand_tkminPixelHitsn,int ChargedHadrCand_tkminTrackerHitsn,double ChargedHadrCand_tkmaxipt,double ChargedHadrCand_tkmaxChi2, Vertex pV);
  PFCandidateRefVector filteredPFChargedHadrCands(PFCandidateRefVector theInitialPFCands,double ChargedHadrCand_tkminPt,int ChargedHadrCand_tkminPixelHitsn,int ChargedHadrCand_tkminTrackerHitsn,double ChargedHadrCand_tkmaxipt,double ChargedHadrCand_tkmaxChi2,double ChargedHadrCand_tktorefpointmaxDZ,Vertex pV, double refpoint_Z);
  PFCandidateRefVector filteredPFNeutrHadrCands(PFCandidateRefVector theInitialPFCands,double NeutrHadrCand_HcalclusMinEt);
  PFCandidateRefVector filteredPFGammaCands(PFCandidateRefVector theInitialPFCands,double GammaCand_EcalclusMinEt);
  math::XYZPoint propagTrackECALSurfContactPoint(const MagneticField*,TrackRef); 

  TFormula computeConeSizeTFormula(const string& ConeSizeFormula,const char* errorMessage);
  void replaceSubStr(string& s,const string& oldSubStr,const string& newSubStr); 

  double computeDeltaR(const math::XYZVector& vec1, const math::XYZVector& vec2);
  double computeAngle(const math::XYZVector& vec1, const math::XYZVector& vec2);
  //binary predicate classes for sorting a list of indexes corresponding to PFRefVectors...(as they can't use STL??)
  class sortRefsByOpeningDistance
  {
     public:
     sortRefsByOpeningDistance(const math::XYZVector& theAxis, double (*ptrToMetricFunction)(const math::XYZVector&, const math::XYZVector&), const PFCandidateRefVector& myInputVector):myMetricFunction(ptrToMetricFunction),axis(theAxis),myVector(myInputVector){};
     bool operator()(uint32_t indexA, uint32_t indexB)
     {
        const PFCandidateRef candA = myVector.at(indexA);
        const PFCandidateRef candB = myVector.at(indexB);
        return (myMetricFunction(axis, candA->momentum()) < myMetricFunction(axis, candB->momentum()));
     }
     private:
     double (*myMetricFunction)(const math::XYZVector&, const math::XYZVector&);
     math::XYZVector axis;  //axis about which candidates are sorted
     const PFCandidateRefVector myVector;
  };
  class filterChargedAndNeutralsByPt
  {
     public:
     filterChargedAndNeutralsByPt(double minNeutralPt, double minChargedPt, const PFCandidateRefVector& myInputVector):minNeutralPt_(minNeutralPt),minChargedPt_(minChargedPt),myVector(myInputVector){};
     bool operator()(uint32_t candIndex)
     {
        const PFCandidateRef cand = myVector.at(candIndex);
        bool output          = true;
        unsigned char charge = abs(cand->charge());
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
     const PFCandidateRefVector& myVector;
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

