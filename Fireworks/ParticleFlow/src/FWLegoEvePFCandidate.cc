#include "Fireworks/ParticleFlow/interface/FWLegoEvePFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

FWLegoEvePFCandidate::FWLegoEvePFCandidate(const reco::PFCandidate& iData):
   m_et(0.f),
   m_pt(0.f)
{
   float base = 0.01; // flor offset

   // first vertical  line , which is et 
   m_et =  iData.et();
   AddLine(iData.eta(),iData.phi(), base, 
           iData.eta(),iData.phi(), base + iData.et());


   AddMarker(0, 1.f);
   SetMarkerStyle(3); 
   SetMarkerSize(0.01); 
   SetDepthTest(false);

   // circle pt
   const unsigned int nLineSegments = 20;
   float circleScalingFactor = 50;
   const double jetRadius = iData.pt()/circleScalingFactor;
  
   m_pt = iData.pt();
   for ( unsigned int iphi = 0; iphi < nLineSegments; ++iphi ) {
      AddLine(iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*iphi),
              iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*iphi),
              base,
              iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*(iphi+1)),
              iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*(iphi+1)),
              base);
   }
}

void
FWLegoEvePFCandidate::UpdateScale(float s)
{
   // resize first line
   TEveChunkManager::iterator li(GetLinePlex());
   li.next();
   TEveStraightLineSet::Line_t& l = * (TEveStraightLineSet::Line_t*) li(); 
   l.fV2[2] = l.fV1[2] + s*m_et;

   // move end point
   TEveChunkManager::iterator mi(GetMarkerPlex());
   mi.next();
   TEveStraightLineSet::Marker_t& m = * (TEveStraightLineSet::Marker_t*) mi();
   m.fV[2] =  l.fV2[2];
}
