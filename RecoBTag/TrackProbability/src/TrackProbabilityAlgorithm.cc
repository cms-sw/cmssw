
#include "RecoBTag/TrackProbability/interface/TrackProbabilityAlgorithm.h"
#include "RecoBTag/BTagTools/interface/SignedTransverseImpactParameter.h"
#include "RecoBTag/BTagTools/interface/SignedImpactParameter3D.h"
#include "RecoBTag/BTagTools/interface/SignedDecayLength3D.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


using namespace std;
using namespace edm;
using namespace reco;


TrackProbabilityAlgorithm::TrackProbabilityAlgorithm() : m_transientTrackBuilder(0),m_probabilityEstimator(0)
{
  m_ipType           = 1;
  m_cutPixelHits     = 2;
  m_cutTotalHits     = 8;
  m_cutMaxTIP        = 1.;
  m_cutMinPt         = 1.;
  m_cutMaxDecayLen   = 8.;
  m_cutMaxChiSquared = 5.;
  m_cutMaxLIP        = 120.;
  m_cutMaxDistToAxis = 0.07;
  m_cutMinProb =0.005;
}

TrackProbabilityAlgorithm::TrackProbabilityAlgorithm(const ParameterSet & parameters) : m_transientTrackBuilder(0),m_probabilityEstimator(0)

{
  m_ipType           = parameters.getParameter<int>("ImpactParamterType");
  m_cutPixelHits     = parameters.getParameter<unsigned int>("MinimumNumberOfPixelHits"); //FIXME: use or remove
  m_cutTotalHits     = parameters.getParameter<unsigned int>("MinimumNumberOfHits"); // used
  m_cutMaxTIP        = parameters.getParameter<double>("MaximumTransverseImpactParameter"); // used
  m_cutMinPt         = parameters.getParameter<double>("MinimumTransverseMomentum"); // used
  m_cutMaxDecayLen   = parameters.getParameter<double>("MaximumDecayLength"); //used
  m_cutMaxChiSquared = parameters.getParameter<double>("MaximumChiSquared"); //used 
  m_cutMaxLIP        = parameters.getParameter<double>("MaximumLongitudinalImpactParameter"); //used
  m_cutMaxDistToAxis = parameters.getParameter<double>("MaximumDistanceToJetAxis"); //used
  m_cutMinProb =  parameters.getParameter<double>("MinimumProbability"); //used

}

pair<JetTag,TrackProbabilityTagInfo> TrackProbabilityAlgorithm::tag(const  JetTracksAssociationRef & jetTracks, const Vertex & pv) 
{

 if(m_probabilityEstimator == 0) 
  {
  edm::LogError ("TrackProbability|BadSetup")    << "Probability estimator is 0. abort!" ;
     abort(); //FIXME: trow an exception here 
  }
 if(m_transientTrackBuilder == 0) 
  {
  edm::LogError ("TrackProbability|BadSetup")     << "Transient track builder is 0. abort!" ;
     abort(); //FIXME: trow an exception here 
  }

 SignedImpactParameter3D  sip3D;  //(m_magneticField);
 SignedTransverseImpactParameter stip;
 multimap<double,int> probability3DMap;
 multimap<double,int> probability2DMap;
 GlobalVector direction(jetTracks->first->px(),jetTracks->first->py(),jetTracks->first->pz());
 double pvZ=pv.z();
 
 edm::RefVector<reco::TrackCollection> tracks=jetTracks->second;

 int i=0; //everything is based on indices
 for(edm::RefVector<reco::TrackCollection>::const_iterator it=tracks.begin() ; it!=tracks.end(); it++ , i++ )
        {
             const Track & track = **it;
             const TransientTrack transientTrack = (m_transientTrackBuilder->build(&(**it)));
             float distToAxis = SignedImpactParameter3D::distanceWithJetAxis(transientTrack,direction,pv).second.value();
             float dLen = SignedDecayLength3D::apply(transientTrack,direction,pv).second.value(); 
               if( track.pt() > m_cutMinPt  &&  //minimum pt
                 fabs(track.d0()) < m_cutMaxTIP && // max transverse i.p.
                 track.recHitsSize() >= m_cutTotalHits && // min num tracker hits
                 fabs(track.dz()-pvZ) < m_cutMaxLIP &&  // z-impact parameter
                 track.normalizedChi2() < m_cutMaxChiSquared &&  // normalized chi2
                 fabs(distToAxis) < m_cutMaxDistToAxis &&
                 fabs(dLen) < m_cutMaxDecayLen
                )
             {
              pair<bool,double> prob3d =  m_probabilityEstimator->probability(0,sip3D.apply(transientTrack,direction,pv).second.significance(),track,*(jetTracks->first),pv);
              pair<bool,double> prob2d =  m_probabilityEstimator->probability(1,stip.apply(transientTrack,direction,pv).second.significance(),track,*(jetTracks->first),pv);
              
              if(prob3d.first)
               { 
                 double p3d;
                 if (prob3d.second >=0){p3d=prob3d.second/2.;}else{p3d=1.+prob3d.second/2.;}
                 if(-log(p3d)> 5) p3d=exp(-5.0);  //FIXME:configurable!!
                 probability3DMap.insert( pair<double,int>(p3d,i));
               }
              if(prob2d.first)
               {
                 double p2d;
                 if (prob2d.second >=0){p2d=prob2d.second/2.;}else{p2d=1.+prob2d.second/2.;}
                 if(-log(p2d)> 5) p2d=exp(-5.0);  //FIXME:configurable!!
                 probability2DMap.insert( pair<double,int>(p2d,i));
               }
             }
         }
 
  vector<double> probability3D,probability2D;
    vector<int> trackOrder3D,trackOrder2D;
  
    for(multimap<double,int>::reverse_iterator it = probability3DMap.rbegin();it != probability3DMap.rend(); it++)
    {
     probability3D.push_back((*it).first); //probability vector
     trackOrder3D.push_back((*it).second);
    }
    
    for(multimap<double,int>::reverse_iterator it = probability2DMap.rbegin();it != probability2DMap.rend(); it++)
    {
     probability2D.push_back((*it).first); //probability vector
     trackOrder2D.push_back((*it).second);
    }

TrackProbabilityTagInfo resultExtended(probability2D,probability3D,trackOrder2D,trackOrder3D,jetTracks);
JetTag resultBase(resultExtended.discriminator(m_ipType,m_cutMinProb));
return pair<JetTag,TrackProbabilityTagInfo> (resultBase,resultExtended); 
}


