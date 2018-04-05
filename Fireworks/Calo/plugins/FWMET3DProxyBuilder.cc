// -*- C++ -*-
//
// Package: Fireworks
// Class  : FWMET3DProxyBuilder.cc

/*

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author: 
//         Created: Mon Jan 17 10:48:11 2011 
//
//

// system include files

// user include files

#include "TMath.h"
#include "TEveArrow.h"
#include "TEveScalableStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "DataFormats/METReco/interface/MET.h"

// forward declarations

class FWMET3DProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::MET> {
public:
   class Arrow : public TEveArrow
   {
   public:
      float m_et;
      float m_energy;
      const FWViewContext* m_vc;

      Arrow(Float_t x, Float_t y, Float_t z,
            Float_t xo, Float_t yo, Float_t zo=0) : 
         TEveArrow(x, y, z, xo, yo, zo),
         m_et(0), m_energy(0), m_vc(nullptr) {}

      void setScale(FWViewEnergyScale* caloScale)
      {
         static float maxW = 3;
         float scale = caloScale->getScaleFactor3D()*(caloScale->getPlotEt() ? m_et : m_energy);
         fVector.Normalize();
         fVector *= scale;
         fTubeR = TMath::Min(maxW/scale, 0.08f);
         fConeR = TMath::Min(maxW*2.5f/scale, 0.25f);

      }
   };

   FWMET3DProxyBuilder();
   ~FWMET3DProxyBuilder() override;

   // ---------- const member functions ---------------------

   bool havePerViewProduct(FWViewType::EType) const override { return true; } // used energy scaling
   void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc) override;
   void cleanLocal() override { m_arrows.clear(); }

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMET3DProxyBuilder(const FWMET3DProxyBuilder&) = delete; // stop default
   const FWMET3DProxyBuilder& operator=(const FWMET3DProxyBuilder&) = delete; // stop default
   
   using FWSimpleProxyBuilderTemplate<reco::MET>::build;
   void build(const reco::MET&, unsigned int, TEveElement&, const FWViewContext*) override;

   // ---------- member data --------------------------------
   std::vector<Arrow*> m_arrows;
};

//
// constructors and destructor
//
FWMET3DProxyBuilder::FWMET3DProxyBuilder()
{
}

FWMET3DProxyBuilder::~FWMET3DProxyBuilder()
{
}

//
// member functions
//
void
FWMET3DProxyBuilder::scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc)
{
   // printf("scale prod \n");
   FWViewEnergyScale* caloScale = vc->getEnergyScale();  

   for (std::vector<Arrow*>::iterator i = m_arrows.begin(); i!= m_arrows.end(); ++ i)
   {
      if ( vc == (*i)->m_vc)
      {
         (*i)->setScale(caloScale);  
      }
   }
}

void 
FWMET3DProxyBuilder::build(const reco::MET& met, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* vc)
{
   float r0;
   float phi   = met.phi();
   float theta = met.theta();

   if (TMath::Abs(met.eta()) < context().caloTransEta())
      r0  = context().caloR1()/sin(theta);
   else
      r0  = context().caloZ1()/fabs(cos(theta));

   Arrow* arrow = new Arrow( sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta),
                             r0*sin(theta)*cos(phi), r0*sin(theta)*sin(phi), r0*cos(theta));
   arrow->m_et = met.et();
   arrow->m_energy = met.energy();
   arrow->m_vc = vc;
   arrow->SetConeL(0.15);
   arrow->SetConeR(0.06);
   setupAddElement(arrow, &oItemHolder );  

   m_arrows.push_back(arrow);
   arrow->setScale(vc->getEnergyScale()); 
   arrow->setScale(vc->getEnergyScale());

   context().voteMaxEtAndEnergy(met.et(), met.energy());

}

//
// const member functions
//

//
// static member functions
//

REGISTER_FWPROXYBUILDER(FWMET3DProxyBuilder, reco::MET, "recoMET", FWViewType::kAll3DBits);
