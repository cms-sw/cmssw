#ifndef ImpactParameter_PromptTrackCountingComputer_h
#define ImpactParameter_PromptTrackCountingComputer_h

// This returns a discriminator equal to the number of prompt tracks in the jet
// It is intended for exotica physics, not b tagging.
// It closely resembles the TrackCountingComputer, but with a different discrinator definition and slightly different cuts.
// Author: Ian Tomalin

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "Math/GenVector/VectorUtil.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

class PromptTrackCountingComputer : public JetTagComputer
{
 public:
  PromptTrackCountingComputer(const edm::ParameterSet  & parameters )
  {
     m_nthTrack         = parameters.getParameter<int>("nthTrack");
     m_ipType           = parameters.getParameter<int>("impactParameterType");
     // Maximum and minimum allowed deltaR respectively. 
     m_deltaR           = parameters.getParameter<double>("deltaR");
     m_deltaRmin      = parameters.getParameter<double>("deltaRmin");
     maxImpactParameter    = parameters.getParameter<double>("maxImpactParameter");
     maxImpactParameterSig = parameters.getParameter<double>("maxImpactParameterSig");
     m_cutMaxDecayLen   = parameters.getParameter<double>("maximumDecayLength"); //used
     m_cutMaxDistToAxis = parameters.getParameter<double>("maximumDistanceToJetAxis"); //used
     //
     // access track quality class; "any" takes everything
     //
     std::string trackQualityType = parameters.getParameter<std::string>("trackQualityClass"); //used
     m_trackQuality =  reco::TrackBase::qualityByName(trackQualityType);
     m_useAllQualities = false;
     if (trackQualityType == "any" || 
	 trackQualityType == "Any" || 
	 trackQualityType == "ANY" ) m_useAllQualities = true;

     uses("ipTagInfos");
  }
  
  float discriminator(const TagInfoHelper & ti) const 
   {
     const reco::TrackIPTagInfo & tkip = ti.get<reco::TrackIPTagInfo>();
     std::multiset<float> significances = orderedSignificances(tkip);
     std::multiset<float>::iterator sig;
     unsigned int nPromptTrk = 0;
     for(sig=significances.begin(); sig!=significances.end(); sig++) {
       if (fabs(*sig) < maxImpactParameterSig) nPromptTrk++;
       //       edm::LogDebug("") << "Track "<< nPromptTrk << " sig=" << *sig;       
     }
     return double(nPromptTrk);
   }

 protected:
     std::multiset<float> orderedSignificances(const reco::TrackIPTagInfo & tkip)   const  {

          const std::vector<reco::TrackIPTagInfo::TrackIPData> & impactParameters((tkip.impactParameterData()));
          const edm::RefVector<reco::TrackCollection> & tracks(tkip.selectedTracks());
          std::multiset<float> significances;
          int i=0;
          if(tkip.primaryVertex().isNull())  {  return std::multiset<float>();}

          GlobalPoint pv(tkip.primaryVertex()->position().x(),tkip.primaryVertex()->position().y(),tkip.primaryVertex()->position().z());

          for(std::vector<reco::TrackIPTagInfo::TrackIPData>::const_iterator it = impactParameters.begin(); it!=impactParameters.end(); ++it, i++)
           {
           if(   fabs(impactParameters[i].distanceToJetAxis.value()) < m_cutMaxDistToAxis  &&        // distance to JetAxis
                 (impactParameters[i].closestToJetAxis - pv).mag() < m_cutMaxDecayLen  &&      // max decay len
		 (m_useAllQualities  == true || (*tracks[i]).quality(m_trackQuality)) // use selected track qualities
             )
	     {
	       if ( ( m_deltaR    <=0  || ROOT::Math::VectorUtil::DeltaR((*tkip.jet()).p4().Vect(), (*tracks[i]).momentum()) < m_deltaR ) &&
	            ( m_deltaRmin <=0  || ROOT::Math::VectorUtil::DeltaR((*tkip.jet()).p4().Vect(), (*tracks[i]).momentum()) > m_deltaRmin ) ) {
		 if ( fabs(((m_ipType==0)?it->ip3d:it->ip2d).value()) < maxImpactParameter ) {
                     significances.insert( ((m_ipType==0)?it->ip3d:it->ip2d).significance() );
                 }
               }
             }
          }
 
         return significances;    
   }
    
   int m_nthTrack;
   int m_ipType;
   double m_deltaR;
   double m_deltaRmin;
   double maxImpactParameter;
   double maxImpactParameterSig;
   double  m_cutMaxDecayLen;
   double m_cutMaxDistToAxis;
   reco::TrackBase::TrackQuality   m_trackQuality;
   bool m_useAllQualities;
};

#endif // ImpactParameter_PromptTrackCountingComputer_h
