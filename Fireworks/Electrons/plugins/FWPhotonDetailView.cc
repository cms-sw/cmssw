//
// Package:     Calo
// Class  :     FWPhotonDetailView
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWPhotonDetailView.cc,v 1.10 2009/07/15 18:58:31 amraktad Exp $

#include "TLatex.h"

#include "Fireworks/Electrons/plugins//FWPhotonDetailView.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

//
// constructors and destructor
//
FWPhotonDetailView::FWPhotonDetailView()
{
}

FWPhotonDetailView::~FWPhotonDetailView()
{
}

//
// member functions
//
void FWPhotonDetailView::build (const FWModelId &id, const reco::Photon* iPhoton, TEveWindowSlot* slot)
{
   if(0==iPhoton) return;
   m_item = id.item();

   FWECALDetailView<reco::Photon>::build_projected(id, iPhoton, slot);
}

void FWPhotonDetailView::makeLegend (const reco::Photon &photon, TCanvas* textCanvas)
{
   TLatex* latex = new TLatex(0.02, 0.970, "");
   latex->SetTextSize(0.06);

   float_t x = 0.02;
   float_t x2 = 0.52;
   float   y = 0.95;
   float fontsize = latex->GetTextSize()*0.6;

   // summary
   char summary[128];
   sprintf(summary, "%s = %.1f GeV",
           "E_{T}", photon.et());
   latex->DrawLatex(x, y, summary);
   y -= fontsize;

   // E/p, H/E
   char hoe[128];
   sprintf(hoe, "H/E = %.3f", photon.hadronicOverEm());
   latex->DrawLatex(x, y, hoe);
   y -= fontsize;

   // phi eta
   char ephi[30];
   sprintf(ephi, " #eta = %.2f", photon.eta());
   latex->DrawLatex(x, y, ephi);
   sprintf(ephi, " #varphi = %.2f", photon.phi());
   latex->DrawLatex(x2, y, ephi);
   y -= 2*fontsize;
   // legend
   latex->DrawLatex(x, y, "#color[2]{#bullet} seed cluster centroid");
   y -= fontsize;
   latex->DrawLatex(x, y, "#color[4]{#bullet} supercluster centroid");
   y -= fontsize;
   latex->DrawLatex(x, y, "#color[618]{#Box} seed cluster");
   y -= fontsize;
   latex->DrawLatex(x, y, "#color[608]{#Box} other clusters"); // kCyan+1
   // eta, phi axis or x, y axis?
   assert(photon.superCluster().isNonnull());
   bool is_endcap = false;
   if (photon.superCluster()->hitsAndFractions().size() > 0 &&
       photon.superCluster()->hitsAndFractions().begin()->first.subdetId() == EcalEndcap)
      is_endcap = true;
}

REGISTER_FWDETAILVIEW(FWPhotonDetailView);
