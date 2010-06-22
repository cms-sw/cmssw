#include "TEveCaloData.h"

#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/ParticleFlow/src/FWPFScale.h"
#include "Fireworks/ParticleFlow/interface/FWLegoEvePFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

FWLegoEvePFCandidate::FWLegoEvePFCandidate(const reco::PFCandidate& iData, const FWViewContext* vc,  const fireworks::Context& context):
   m_energy(0.f),
   m_et(0.f)
{
   m_et =  iData.et();
   m_energy = iData.energy();

  
   // energy auto scale  
   FWViewEnergyScale* scaleE = vc->getEnergyScale("PFenergy");
   if (!scaleE)
   {
      scaleE = new FWPFScale();
      vc->addScale("PFenergy", scaleE);
   }
   scaleE->setVal(m_energy);

   //et auto scale
   FWViewEnergyScale* scaleEt = vc->getEnergyScale("PFet");
   if (!scaleEt)
   {
      scaleEt = new FWPFScale();
      vc->addScale("PFet", scaleEt);
   }
   scaleEt->setVal(m_et);
   

   float base = 0.01; // flour offset 1%

   // first vertical  line , which is et/energy
   float val = vc->getPlotEt() ?  m_et : m_energy;
   AddLine(iData.eta(),iData.phi(), base, 
           iData.eta(),iData.phi(), base + val*getScale(vc, context));


   AddMarker(0, 1.f);
   SetMarkerStyle(3); 
   SetMarkerSize(0.01); 
   SetDepthTest(false);

   // circle pt
   const unsigned int nLineSegments = 20;
   float circleScalingFactor = 50;
   const double jetRadius = iData.pt()/circleScalingFactor;
  
   for ( unsigned int iphi = 0; iphi < nLineSegments; ++iphi ) {
      AddLine(iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*iphi),
              iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*iphi),
              base,
              iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*(iphi+1)),
              iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*(iphi+1)),
              base);
   }
}

float
FWLegoEvePFCandidate::getScale(const FWViewContext* vc, const fireworks::Context& context) const
{
   float s = 1.f;
   if (context.getCaloData()->Empty()  && vc->getAutoScale())
   {
       if (vc->getPlotEt())
      {
         s = vc->getEnergyScale("PFet")->getVal();
      }
      else
      {
         s = vc->getEnergyScale("PFenergy")->getVal();
      }
   }
   else
   {
      // calorimeter TEveCaloLego scale
      s = vc->getEnergyScale("Calo")->getVal();
   }

   return s*TMath::Pi();
}

void
FWLegoEvePFCandidate::updateScale(const FWViewContext* vc, const fireworks::Context& context)
{
   float val = vc->getPlotEt() ?  m_et : m_energy;

   // resize first line
   TEveChunkManager::iterator li(GetLinePlex());
   li.next();
   TEveStraightLineSet::Line_t& l = * (TEveStraightLineSet::Line_t*) li(); 
   l.fV2[2] = l.fV1[2] + val*getScale(vc, context);

   // move end point
   TEveChunkManager::iterator mi(GetMarkerPlex());
   mi.next();
   TEveStraightLineSet::Marker_t& m = * (TEveStraightLineSet::Marker_t*) mi();
   m.fV[2] =  l.fV2[2];
}

