// -*- C++ -*-
// $Id: HCalCaloTowerProxy3DBuilder.cc,v 1.7 2008/11/10 18:07:57 amraktad Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "RVersion.h"
#include "TEveCalo.h"
#include "TEveCaloData.h"
#include "TH2F.h"
#include "Fireworks/Calo/interface/HCalCaloTowerProxy3DBuilder.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"


// user include files
/*
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

#include "Fireworks/Calo/interface/HCalCaloTowerProxy3DBuilder.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"

void HCalCaloTowerProxy3DBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   if( *product == 0) *product = new TEveElementList();
   TH2F* hist = 0;
   TEveCaloDataHist* data = 0;
   std::string name = "hcal3D";
   if ( m_calo3d ) data = dynamic_cast<TEveCaloDataHist*>( m_calo3d->GetData() );
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
     hist = new TH2F(name.c_str(),"CaloTower HCAL Et distribution", 82, fw3dlego::xbins, 72, -M_PI, M_PI);
     TH1::AddDirectory(status);
     newHist = true;
   }
   hist->Reset();
   for(CaloTowerCollection::const_iterator tower = towers->begin(); tower != towers->end(); ++tower)
     (hist)->Fill(tower->eta(), tower->phi(), tower->hadEt()+tower->outerEt());

   if ( ! data ) data = new TEveCaloDataHist();
   if ( newHist ) {
      Int_t s = data->AddHistogram(hist);
      data->RefSliceInfo(s).Setup("HCAL", 0., iItem->defaultDisplayProperties().color());
   }

   if ( m_calo3d == 0 ) {
      m_calo3d = new TEveCalo3D(data);
      m_calo3d->SetBarrelRadius(129);
      m_calo3d->SetEndCapPos(310);
      // gEve->AddElement(m_calo3d);
      //	(*product)->AddElement(m_calo3d);
      gEve->AddElement(m_calo3d, *product);
   }
}
*/
std::string
HCalCaloTowerProxy3DBuilder::histName() const
{
   return "hcal3D";
}

REGISTER_FWRPZDATAPROXYBUILDERBASE(HCalCaloTowerProxy3DBuilder,CaloTowerCollection,"HCal");
