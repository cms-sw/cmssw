// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloTauProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWCaloTauProxyBuilder.cc,v 1.17 2012/03/23 00:08:29 amraktad Exp $
//

// system include files
#include "TEveCompound.h"
#include "TGeoTube.h"
#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"
#include "TEveTrack.h"

// user include files
#include "Fireworks/Calo/interface/FWTauProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "Fireworks/Calo/interface/thetaBins.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauFwd.h"

class FWViewContext;

class FWCaloTauProxyBuilder : public FWTauProxyBuilderBase
{
public:
   FWCaloTauProxyBuilder() {}
   virtual ~FWCaloTauProxyBuilder() {}
  
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCaloTauProxyBuilder(const FWCaloTauProxyBuilder&);    // stop default
   const FWCaloTauProxyBuilder& operator=(const FWCaloTauProxyBuilder&);    // stop default

   virtual void buildViewType( const FWEventItem* iItem, TEveElementList* product, FWViewType::EType viewType , const FWViewContext* vc);

};

void
FWCaloTauProxyBuilder::buildViewType( const FWEventItem* iItem, TEveElementList* product, FWViewType::EType viewType , const FWViewContext* vc)
{
   reco::CaloTauCollection const * caloTaus = 0;
   iItem->get( caloTaus );
   if( caloTaus == 0 ) return;

      
   Int_t idx = 0;
   for( reco::CaloTauCollection::const_iterator it = caloTaus->begin(), itEnd = caloTaus->end(); it != itEnd; ++it, ++idx)
   {  
      TEveCompound* comp = createCompound();

      if (viewType == FWViewType::kLego)
      {
         fireworks::addCircle( (*it).eta(), (*it).phi(), 0.5, 20, comp, this );
      }
      else
      {
         try {
            const reco::CaloTauTagInfo *tauTagInfo = dynamic_cast<const reco::CaloTauTagInfo*>(((*it).caloTauTagInfoRef().get()));
            const reco::CaloJet *jet = dynamic_cast<const reco::CaloJet*>((tauTagInfo->calojetRef().get()));

            int min =  100;
            int max = -100;
            std::vector<double> phis;
            std::vector<CaloTowerPtr> towers = jet->getCaloConstituents();
            for( std::vector<CaloTowerPtr>::const_iterator tower = towers.begin(), towerEnd = towers.end();
                 tower != towerEnd; ++tower )
            {
               unsigned int ieta = 41 + (*tower)->id().ieta();
               if( ieta > 40 ) --ieta;
               assert( ieta <= 82 );
	 
               if( int(ieta) > max ) max = ieta;
               if( int(ieta) < min ) min = ieta;
               m_phis.push_back( (*tower)->phi() );
            }
            if( min > max ) {	
               min = 0; max = 0;
            }
            const std::vector<std::pair<double, double> > thetaBins = fireworks::thetaBins();
            m_minTheta = thetaBins[min].first;
            m_maxTheta = thetaBins[max].second;

            buildBaseTau(*it, jet, comp, viewType, vc);
            m_phis.clear();
         }  
         catch (std::exception&  e)
         { 
            fwLog(fwlog::kInfo) << "FWPFTauProxyBuilder missing PFTauTagInfo. Skip drawing of jets.\n";
            buildBaseTau(*it, 0, comp, viewType, vc);
         }
      }
      setupAddElement( comp, product );
   }
}


REGISTER_FWPROXYBUILDER(FWCaloTauProxyBuilder, reco::CaloTauCollection, "CaloTau", FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
