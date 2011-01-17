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
      TEveScalableStraightLineSet* m_line;

      Arrow(Float_t x, Float_t y, Float_t z,
            Float_t xo, Float_t yo, Float_t zo=0) : 
         TEveArrow(x, y, z, xo, yo, zo),
         m_et(0), m_energy(0), m_vc(0), m_line(0) {}

      void setScale(FWViewEnergyScale* caloScale)
      {
         static float maxW = 3;
         float scale = caloScale->getScaleFactor3D()*(caloScale->getPlotEt() ? m_et : m_energy);
         // printf("scale arroe %f \n", scale);
         fVector.Normalize();
         fVector *= scale;
         float w = 0.02/scale;
         fTubeR = TMath::Min(maxW/scale, 0.02f);
         fConeR = fTubeR*2;

         m_line->SetScale(scale);
      }
   };

   FWMET3DProxyBuilder();
   virtual ~FWMET3DProxyBuilder();

   // ---------- const member functions ---------------------

   virtual bool havePerViewProduct(FWViewType::EType) const { return true; } // used energy scaling
   virtual void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc);
   virtual void cleanLocal() { m_arrows.clear(); }

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMET3DProxyBuilder(const FWMET3DProxyBuilder&); // stop default
   const FWMET3DProxyBuilder& operator=(const FWMET3DProxyBuilder&); // stop default
   
   void build(const reco::MET&, unsigned int, TEveElement&, const FWViewContext*);

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
   setupAddElement(arrow, &oItemHolder );
  
   // draw line in case of zoom out;
   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet( "MET marker" );
   marker->SetLineWidth( 2 );
   marker->SetScaleCenter( r0*sin(theta)*cos(phi), r0*sin(theta)*sin(phi), r0*cos(theta) );
   float r1 = r0 + 1;
   marker->AddLine( r0*sin(theta)*cos(phi), r0*sin(theta)*sin(phi), r0*cos(theta),
                    r1*sin(theta)*cos(phi), r1*sin(theta)*sin(phi), r1*cos(theta) );
   setupAddElement( marker, &oItemHolder );

   arrow->m_line = marker;
   m_arrows.push_back(arrow);
   arrow->setScale(vc->getEnergyScale());
}

//
// const member functions
//

//
// static member functions
//

REGISTER_FWPROXYBUILDER(FWMET3DProxyBuilder, reco::MET, "recoMET", FWViewType::kAll3DBits);
