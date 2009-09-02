
#include "Fireworks/Calo/interface/FWECALDetailViewBuilder.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "TAxis.h"
#include "TGLViewer.h"
#include "TEveCaloLegoOverlay.h"

// this is temporary until a fix is made in root
#define protected public
#include "TEveLegoEventHandler.h"
#undef protected

#include "TGeoMatrix.h"
#include "TEveTrans.h"

TEveCaloLego* FWECALDetailViewBuilder::build()
{
		
	// get the hits from the event
	
	fwlite::Handle<EcalRecHitCollection> handle_hits;
	const EcalRecHitCollection *hits = 0;
	
	if (fabs(m_eta) < 1.5) {
		try {
			handle_hits.getByLabel(*m_event, "ecalRecHit", "EcalRecHitsEB");
			hits = handle_hits.ptr();
		}
		catch (...)
		{
			std::cout <<"no barrel ECAL rechits are available, "
			"showing crystal location but not energy" << std::endl;
		}	
	} else {
		try {
			handle_hits.getByLabel(*m_event, "ecalRecHit", "EcalRecHitsEE");
			hits = handle_hits.ptr();
		}
		catch (...)
		{
			std::cout <<"no endcap ECAL rechits are available, "
			"showing crystal location but not energy" << std::endl;
		}
	} 
	
	// data
	TEveCaloDataVec* data = new TEveCaloDataVec(1 + m_colors.size());
	data->RefSliceInfo(0).Setup("hits (not clustered)", 0.0, kMagenta+2);   
	for (size_t i = 0; i < m_colors.size(); ++i)
	{
		data->RefSliceInfo(i + 1).Setup("hits (not clustered)", 0.0, m_colors[i]);   
	}
	
	// fill
	fillData(hits, data);	
	
	// make grid
	Double_t em, eM, pm, pM;
	data->GetEtaLimits(em, eM);
	data->GetPhiLimits(pm, pM);
	data->SetAxisFromBins((eM-em)*0.05, (pM-pm)*0.05); // 5% percision
	if (fabs(m_eta) > 1.5) {
		data->GetEtaBins()->SetTitle("X[cm]");
		data->GetPhiBins()->SetTitle("Y[cm]");
	} else {
		data->GetEtaBins()->SetTitleFont(122);
		data->GetEtaBins()->SetTitle("h");
		data->GetPhiBins()->SetTitleFont(122);
		data->GetPhiBins()->SetTitle("f");
	}
	
	// lego
	TEveCaloLego *lego = new TEveCaloLego(data);
	// scale and translate to real world coordinates
	lego->SetEta(em, eM);
	lego->SetPhiWithRng((pm+pM)*0.5, (pM-pm)*0.5); // phi range = 2* phiOffset
	Double_t legoScale = ((eM - em) < (pM - pm)) ? (eM - em) : (pM - pm);
	lego->InitMainTrans();
	lego->RefMainTrans().SetScale(legoScale, legoScale, legoScale*0.5);
	lego->RefMainTrans().SetPos((eM+em)*0.5, (pM+pm)*0.5, 0);
	lego->SetAutoRebin(kFALSE);
	lego->Set2DMode(TEveCaloLego::kValSize);
	lego->SetProjection(TEveCaloLego::kAuto);
	lego->SetName("ECALDetail Lego");
	lego->SetFontColor(kGray);
		
	return lego;
	
}

void FWECALDetailViewBuilder::setColor(Color_t color, const std::vector<DetId> &detIds)
{
	
	m_colors.push_back(color);
	
	// get the slice for this group of detIds
	// note that the zeroth slice is the default one (all else)
	int slice = m_colors.size();
	
	// take a note of which slice these detids are going to go into
	for (size_t i = 0; i < detIds.size(); ++i)
	{
		m_detIdsToColor.insert(std::make_pair<DetId, int>(detIds[i], slice));
	}
							  
}

void FWECALDetailViewBuilder::showSuperClusters(Color_t color)
{

	// get the superclusters from the event

	fwlite::Handle<reco::SuperClusterCollection> handle_superclusters;
   	const reco::SuperClusterCollection *superclusters = 0;
	
	if (fabs(m_eta) < 1.5) {
		try {
			handle_superclusters.getByLabel(*m_event, "correctedHybridSuperClusters");
			superclusters = handle_superclusters.ptr();
		}
		catch ( ...)
		{
			std::cout <<"no barrel superclusters are available" << std::endl;
		}		
	} else {
		try {
			handle_superclusters.getByLabel(*m_event, "correctedMulti5x5SuperClustersWithPreshower");
			superclusters = handle_superclusters.ptr();
		}
		catch ( ...)
		{
			std::cout <<"no endcap superclusters are available" << std::endl;
		}	
	} 	
	
	// set the colors for the super clusters
	std::vector<DetId> scDetIds;
	for (size_t i = 0; i < superclusters->size(); ++i)
	{
		scDetIds.clear();
		const std::vector<std::pair<DetId, float> > &hitsAndFractions = (*superclusters)[i].hitsAndFractions();
		for (size_t j = 0; j < hitsAndFractions.size(); ++j)
		{
			scDetIds.push_back(hitsAndFractions[j].first);
		}
		setColor(color + i, scDetIds);
	}
	
}

void FWECALDetailViewBuilder::fillData (const EcalRecHitCollection *hits, 
										TEveCaloDataVec *data)
{
	
	 // loop on all the detids
	 for (EcalRecHitCollection::const_iterator k = hits->begin();
		  k != hits->end(); ++k) {
	
		 const TGeoHMatrix *matrix = m_item->getGeom()->getMatrix(k->id().rawId());
		 if ( matrix == 0 ) {
			 printf("Warning: cannot get geometry for DetId: %d. Ignored.\n",k->id().rawId());
			 continue;
		 }
	 
		 TVector3 v(matrix->GetTranslation()[0],
					matrix->GetTranslation()[1],
					matrix->GetTranslation()[2]);
	 
		 // set the et
		 double size = k->energy()/cosh(v.Eta());
		 
		 // check what slice to put in
		 int slice = 0;
		 std::map<DetId, int>::const_iterator itr = m_detIdsToColor.find(k->id());
		 if (itr != m_detIdsToColor.end()) slice = itr->second;

		// if in the EB
		 if (k->id().subdetId() == EcalBarrel) {
			 
			 // do phi wrapping
			 double phi = v.Phi();
			 if (v.Phi() > m_phi + M_PI)
			    phi -= 2 * M_PI;
			 if (v.Phi() < m_phi - M_PI)
			    phi += 2 * M_PI;
			 
			 // check if the hit is in the window to be drawn
			 if (! (fabs(v.Eta() - m_eta) < (m_size*0.0172) 
					&& fabs(phi - m_phi) < (m_size*0.0172)))
				 continue;
			 
			 // if in the window to be drawn then draw it
			 data->AddTower(v.Eta() - 0.0172 / 2, v.Eta() + 0.0172 / 2,
							phi - 0.0172 / 2, phi + 0.0172 / 2);
			 data->FillSlice(slice, size);
			 
		// otherwise in the EE
		 } else if (k->id().subdetId() == EcalEndcap) {
			 			 
			 // check if the hit is in the window to be drawn
			 if (! (fabs(v.Eta() - m_eta) < (m_size*0.0172) 
					&& fabs(v.Phi() - m_phi) < (m_size*0.0172)))
				 continue;			 
			 
			 // if in the window to be drawn then draw it
			 data->AddTower((v.X() - 2.9 / 2), (v.X() + 2.9 / 2),
							(v.Y() - 2.9 / 2), (v.Y() + 2.9 / 2));
			 data->FillSlice(slice, size);
		 }
	 
	} // end loop on hits
	 
	 data->DataChanged();
	
}

