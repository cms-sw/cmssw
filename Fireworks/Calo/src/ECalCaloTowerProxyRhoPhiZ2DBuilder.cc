// -*- C++ -*-
// $Id: ECalCaloTowerProxyRhoPhiZ2DBuilder.cc,v 1.20 2008/11/26 16:19:12 chrjones Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TGeoArb8.h"
#include "TEveManager.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TColor.h"
#include "TROOT.h"
#include "TEveCalo.h"
#include "TEveCaloData.h"

#include <iostream>

// user include files
#include "Fireworks/Calo/interface/ECalCaloTowerProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWDisplayEvent.h"

#include "Fireworks/Core/interface/FWRhoPhiZView.h"
#include "Fireworks/Core/interface/fw3dlego_xbins.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ECalCaloTowerProxyRhoPhiZ2DBuilder::ECalCaloTowerProxyRhoPhiZ2DBuilder()
{
   setHighPriority( true );
}

// ECalCaloTowerProxyRhoPhiZ2DBuilder::ECalCaloTowerProxyRhoPhiZ2DBuilder(const ECalCaloTowerProxyRhoPhiZ2DBuilder& rhs)
// {
//    // do actual copying here;
// }

ECalCaloTowerProxyRhoPhiZ2DBuilder::~ECalCaloTowerProxyRhoPhiZ2DBuilder()
{
}

//
// member functions
//

void ECalCaloTowerProxyRhoPhiZ2DBuilder::buildCalo(const FWEventItem* iItem,
						   TEveElementList** product,
						   std::string name,
						   TEveCalo3D*& calo3d,
						   bool ecal)
{
   if( *product == 0)  {
      *product = new TEveElementList();
      gEve->AddElement(*product);
   }
   TH2F* hist = 0;
   TEveCaloDataHist* data = 0;
   if ( calo3d ) data = dynamic_cast<TEveCaloDataHist*>( calo3d->GetData() );
   if ( data ) {
      for ( Int_t i = 0; i < data->GetNSlices(); ++i ){
	 TH2F* h = data->RefSliceInfo(i).fHist;
	 if ( ! h ) continue;
	 if ( name == h->GetName() ) {
	    hist = h;
	    break;
	 }
      }
   }
   const CaloTowerCollection* towers=0;
   iItem->get(towers);
   if(0==towers) return;

   bool newHist = false;
   if ( hist == 0 ) {
      Bool_t status = TH1::AddDirectoryStatus();
      TH1::AddDirectory(kFALSE); //Keeps histogram from going into memory
      hist = new TH2F(name.c_str(),"CaloTower Et distribution", 82, fw3dlego::xbins, 72, -M_PI, M_PI);
      TH1::AddDirectory(status);
      newHist = true;
   }
   hist->Reset();
   if ( ecal )
     for(CaloTowerCollection::const_iterator tower = towers->begin(); tower != towers->end(); ++tower)
       (hist)->Fill(tower->eta(), tower->phi(), tower->emEt());
   else
     for(CaloTowerCollection::const_iterator tower = towers->begin(); tower != towers->end(); ++tower)
       (hist)->Fill(tower->eta(), tower->phi(), tower->hadEt()+tower->outerEt());
   if ( ! data ) data = new TEveCaloDataHist();
   if ( newHist ) {
      Int_t s = data->AddHistogram(hist);
      data->RefSliceInfo(s).Setup(name.c_str(), 0., iItem->defaultDisplayProperties().color());
   }

   if ( calo3d == 0 ) {
      calo3d = new TEveCalo3D(data);
      calo3d->SetBarrelRadius(129);
      calo3d->SetEndCapPos(310);
      // gEve->AddElement(calo3d);
      //	(*product)->AddElement(calo3d);
      gEve->AddElement(calo3d, *product);
   }
}

void
ECalCaloTowerProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem,
						TEveElementList** product)
{
   buildCalo(iItem, product, "ecalRhoPhi", m_caloRhoPhi,true);
}

void
ECalCaloTowerProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem,
					    TEveElementList** product)
{
   buildCalo(iItem, product, "ecalRhoZ", m_caloRhoZ, true);
}

   // NOTE:
   //       Here we assume 72 bins in phi. At high eta we have only 36 and at the
   //       very end 18 bins. These large bins are splited among smaller bins
   //       decreasing energy in each entry by factor of 2 and 4 for 36 and 18 bin
   //       cases. Other options will be implemented later
   //
   // http://ecal-od-software.web.cern.ch/ecal-od-software/documents/documents/cal_newedm_roadmap_v1_0.pdf
   // Eta mapping:
   //   ieta - [-41,-1]+[1,41] - total 82 bins
   //   calo tower gives eta of the ceneter of each bin
   //   size:
   //      0.087 - [-20,-1]+[1,20]
   //      the rest have variable size from 0.09-0.30
   // Phi mapping:
   //   iphi - [1-72]
   //   calo tower gives phi of the center of each bin
   //   for |ieta|<=20 phi bins are all of the same size
   //      iphi 36-37 transition corresponds to 3.1 -> -3.1 transition
   //   for 20 < |ieta| < 40
   //      there are only 36 active bins corresponding to odd numbers
   //      iphi 35->37, corresponds to 3.05 -> -3.05 transition
   //   for |ieta| >= 40
   //      there are only 18 active bins 3,7,11,15 etc
   //      iphi 31 -> 35, corresponds to 2.79253 -> -3.14159 transition

std::vector<std::pair<double,double> >
ECalCaloTowerProxyRhoPhiZ2DBuilder::getThetaBins()
{
   std::vector<std::pair<double,double> > thetaBins(82);
   for ( unsigned int i = 0; i < 82; ++i )
     {
	thetaBins[i].first  = 2*atan( exp(-fw3dlego::xbins[i]) );
	thetaBins[i].second = 2*atan( exp(-fw3dlego::xbins[i+1]) );
     }
   return thetaBins;
}



//
// const member functions
//
REGISTER_FWRPZDATAPROXYBUILDERBASE(ECalCaloTowerProxyRhoPhiZ2DBuilder,CaloTowerCollection,"ECalRhoZSeparate");

//
// static member functions
//
