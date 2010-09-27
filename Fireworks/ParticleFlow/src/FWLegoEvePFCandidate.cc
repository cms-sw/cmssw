#include "TEveCaloData.h"

#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/ParticleFlow/interface/FWLegoEvePFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "Fireworks/Core/interface/fwLog.h"


FWLegoEvePFCandidate::FWLegoEvePFCandidate(const reco::PFCandidate& iData, const FWViewContext* vc,  const fireworks::Context& context):
   m_energy(0.f),
   m_et(0.f)
{
   m_et =  iData.et();
   m_energy = iData.energy();

   // energy auto scale  
   FWViewEnergyScale* scaleE = vc->getEnergyScale("PFenergy");
   scaleE->setMaxVal(m_energy);

   //et auto scale
   FWViewEnergyScale* scaleEt = vc->getEnergyScale("PFet");
   scaleEt->setMaxVal(m_et);

   float base = 0.001; // flour offset 1%

   // first vertical  line , which is et/energy
   FWViewEnergyScale* caloScale = vc->getEnergyScale("Calo");
   float val = caloScale->getPlotEt() ?  m_et : m_energy;
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
   float s = 0.f;
 
   FWViewEnergyScale* caloScale = vc->getEnergyScale("Calo");
   if (context.getCaloData()->Empty()  && caloScale->getScaleMode() == FWViewEnergyScale::kAutoScale)
   {
      // presume plotEt flag is same for "Calo" and particle flow 
      if (caloScale->getPlotEt())
      {
         s = vc->getEnergyScale("PFet")->getMaxVal();
      }
      else
      {
         s = vc->getEnergyScale("PFenergy")->getMaxVal();
      }
      
      // check (if this is used in simple proxy builder than assert will be better)
      if (s == 0.f) {
         fwLog(fwlog::kError) << "FWLegoEvePFCandidate max value is zero !";
         s = 1.f;
      }
      
      // height of TEveCaloLego is TMath::Pi(), see FWLegoViewBase::setContext()
      return TMath::Pi()/s;      
   }
   else
   {
      // height of TEveCaloLego is TMath::Pi(), see FWLegoViewBase::setContext()
      return caloScale->getValToHeight()*TMath::Pi();
   }
}

void
FWLegoEvePFCandidate::updateScale(const FWViewContext* vc, const fireworks::Context& context)
{
   FWViewEnergyScale* caloScale = vc->getEnergyScale("Calo");
   float val = caloScale->getPlotEt() ?  m_et : m_energy;

   // printf("update scale %f \n", getScale(vc, context)); fflush(stdout);
   
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

