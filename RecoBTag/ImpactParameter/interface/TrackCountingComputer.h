#ifndef ImpactParameter_TrackCountingComputer_h
#define ImpactParameter_TrackCountingComputer_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "Math/GenVector/VectorUtil.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

class TrackCountingComputer : public JetTagComputer
{
 public:
  TrackCountingComputer(const edm::ParameterSet  & parameters )
  {
     m_nthTrack         = parameters.getParameter<int>("nthTrack");
     m_ipType           = parameters.getParameter<int>("impactParamterType");
     m_deltaR           = parameters.getParameter<double>("deltaR");
     m_cutMaxDecayLen   = parameters.getParameter<double>("maximumDecayLength"); //used
     m_cutMaxDistToAxis = parameters.getParameter<double>("maximumDistanceToJetAxis"); //used
  }

 
  float discriminator(const reco::BaseTagInfo & ti) const 
   {
    const reco::TrackIPTagInfo * tkip = dynamic_cast<const reco::TrackIPTagInfo *>(&ti);
      if(tkip!=0)  {
          std::multiset<float> significances = orderedSignificances(*tkip);
          std::multiset<float>::reverse_iterator nth=significances.rbegin();
          for(int i=0;i<m_nthTrack-1 && nth!=significances.rend();i++) nth++;  
          if(nth!=significances.rend()) return *nth; else return -100.;
        }
        else 
          {
            //FIXME: report some error? 
            return -100. ;   
          }
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
           if(   fabs(impactParameters[i].distanceToJetAxis) < m_cutMaxDistToAxis  &&        // distance to JetAxis
                 (impactParameters[i].closestToJetAxis - pv).mag() < m_cutMaxDecayLen        // max decay len
             )
              {
                if(m_deltaR <=0  || ROOT::Math::VectorUtil::DeltaR((*tkip.jet()).p4().Vect(), (*tracks[i]).momentum()) < m_deltaR)
                 significances.insert( ((m_ipType==0)?it->ip3d:it->ip2d).significance() );
              }
          }
 
         return significances;    
   }


    
   int m_nthTrack;
   int m_ipType;
   double m_deltaR;
   double  m_cutMaxDecayLen;
   double m_cutMaxDistToAxis;

};

#endif // ImpactParameter_TrackCountingComputer_h
