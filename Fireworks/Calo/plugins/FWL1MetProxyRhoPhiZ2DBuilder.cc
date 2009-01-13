// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWL1MetProxyRhoPhiZ2DBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWL1MetProxyRhoPhiZ2DBuilder.cc,v 1.5 2009/01/13 20:15:46 amraktad Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TGeoArb8.h"
#include "TEveManager.h"
#include "TGeoSphere.h"
#include "TGeoTube.h"
#include "TH1F.h"
#include "TColor.h"
#include "TROOT.h"
#include "TEvePointSet.h"
#include "TEveScalableStraightLineSet.h"
#include "TEveCompound.h"

// user include files
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"

class FWL1MetProxyRhoPhiZ2DBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      FWL1MetProxyRhoPhiZ2DBuilder();
      virtual ~FWL1MetProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      virtual void buildRhoPhi(const FWEventItem* iItem,
                               TEveElementList** product);

      virtual void buildRhoZ(const FWEventItem* iItem,
                               TEveElementList** product);

      double getTheta( double eta ) { return 2*atan(exp(-eta)); }

      FWL1MetProxyRhoPhiZ2DBuilder(const FWL1MetProxyRhoPhiZ2DBuilder&); // stop default

      const FWL1MetProxyRhoPhiZ2DBuilder& operator=(const FWL1MetProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------
};

//
// constructors and destructor
//
FWL1MetProxyRhoPhiZ2DBuilder::FWL1MetProxyRhoPhiZ2DBuilder()
{
}

// FWL1MetProxyRhoPhiZ2DBuilder::FWL1MetProxyRhoPhiZ2DBuilder(const FWL1MetProxyRhoPhiZ2DBuilder& rhs)
// {
//    // do actual copying here;
// }

FWL1MetProxyRhoPhiZ2DBuilder::~FWL1MetProxyRhoPhiZ2DBuilder()
{
}

//
// member functions
//
void
FWL1MetProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem,
					    TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"L1Mets RhoPhi",true);
      *product = tList;
      tList->SetMainColor(   iItem->defaultDisplayProperties().color() );
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }

   // Get the particle map collection for L1EtMissParticles
   l1extra::L1EtMissParticleCollection const * mets=0;
   iItem->get(mets);
   if(0==mets) return;

   double r_ecal = 126;

   fw::NamedCounter counter("met");

   for(unsigned int i = 0; i < mets->size(); ++i, ++counter) {
      const unsigned int nBuffer = 1024;
      char title[nBuffer];
      snprintf(title, nBuffer, "L1 MET: %0.1f GeV", mets->at(i).et());
      TEveCompound* container = new TEveCompound( counter.str().c_str(), title );
      container->OpenCompound();
      //guarantees that CloseCompound will be called no matter what happens
      boost::shared_ptr<TEveCompound> sentry(container,boost::mem_fn(&TEveCompound::CloseCompound));

      double phi = mets->at(i).phi();
      double size = mets->at(i).et();

      TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
      marker->SetLineWidth(1);
      marker->SetLineStyle(2);
      // marker->SetLineStyle(kDotted);
      marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
      marker->SetScaleCenter( r_ecal*cos(phi), r_ecal*sin(phi), 0 );
      marker->AddLine( r_ecal*cos(phi), r_ecal*sin(phi), 0, (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
      marker->AddLine( (r_ecal+size*0.9)*cos(phi+0.01), (r_ecal+size*0.9)*sin(phi+0.01), 0,
		       (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
      marker->AddLine( (r_ecal+size*0.9)*cos(phi-0.01), (r_ecal+size*0.9)*sin(phi-0.01), 0,
		       (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
      container->AddElement(marker);
      container->SetRnrSelf(     iItem->defaultDisplayProperties().isVisible() );
      container->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );

      tList->AddElement(container);
   }
}

void
FWL1MetProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem,
					    TEveElementList** product)
{
   TEveElementList* tList = *product;
   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"L1Mets RhoZ",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }

   // Get the particle map collection for L1EtMissParticles
   l1extra::L1EtMissParticleCollection const * mets=0;
   iItem->get(mets);
   if(0==mets) return;

   double r = 126;

   fw::NamedCounter counter("met");

   for(unsigned int i = 0; i < mets->size(); ++i, ++counter) {
      const unsigned int nBuffer = 1024;
      char title[nBuffer];
      snprintf(title, nBuffer, "L1 MET: %0.1f GeV", mets->at(i).et());
      TEveCompound* container = new TEveCompound( counter.str().c_str(), title );
      container->OpenCompound();
      //guarantees that CloseCompound will be called no matter what happens
      boost::shared_ptr<TEveCompound> sentry(container,boost::mem_fn(&TEveCompound::CloseCompound));

      double phi = mets->at(i).phi();
      double size = mets->at(i).et();

      TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
      marker->SetLineWidth(1);
      marker->SetLineStyle(2);
      marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
      marker->SetScaleCenter(0., (phi>0 ? r : -r), 0);
      marker->AddLine(0., (phi>0 ? r : -r), 1,
		      0., (phi>0 ? (r+size) : -(r+size)), 1 );
      marker->AddLine(0., (phi>0 ? r+size*0.9 : -(r+size*0.9) ), 1+r*0.01,
		      0., (phi>0 ? (r+size) : -(r+size)), 1 );
      marker->AddLine(0., (phi>0 ? r+size*0.9 : -(r+size*0.9) ), 1-r*0.01,
		      0., (phi>0 ? (r+size) : -(r+size)), 1 );
      container->AddElement( marker );
      container->SetRnrSelf(     iItem->defaultDisplayProperties().isVisible() );
      container->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );
      tList->AddElement(container);
   }
}

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWL1MetProxyRhoPhiZ2DBuilder,l1extra::L1EtMissParticleCollection,"L1-MET");
