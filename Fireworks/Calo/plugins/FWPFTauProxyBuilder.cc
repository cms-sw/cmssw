// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWPFTauProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
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
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/BuilderUtils.h"


#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TrackReco/interface/Track.h"

class FWPFTauProxyBuilder : public FWTauProxyBuilderBase
{
public:
   FWPFTauProxyBuilder() {}
   ~FWPFTauProxyBuilder() override {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPFTauProxyBuilder( const FWPFTauProxyBuilder& ) = delete;    // stop default
   const FWPFTauProxyBuilder& operator=( const FWPFTauProxyBuilder& ) = delete;    // stop default

   using FWTauProxyBuilderBase::buildViewType;
   void buildViewType( const FWEventItem* iItem, TEveElementList* product, FWViewType::EType type , const FWViewContext*) override;
};

void
FWPFTauProxyBuilder::buildViewType( const FWEventItem* iItem, TEveElementList* product, FWViewType::EType viewType , const FWViewContext* vc)
{
   reco::PFTauCollection const * pfTaus = nullptr;
   iItem->get( pfTaus );
   if( pfTaus == nullptr ) return;

   for( reco::PFTauCollection::const_iterator it = pfTaus->begin(), itEnd = pfTaus->end(); it != itEnd; ++it)
   { 
      TEveCompound* comp = createCompound();
      if (viewType == FWViewType::kLego)
      {
         fireworks::addCircle( (*it).eta(), (*it).phi(), 0.5, 20, comp, this );
      }
      else
      {
         // prepare phi-list and theta range
         try {
            const reco::PFTauTagInfo *tauTagInfo = dynamic_cast<const reco::PFTauTagInfo*>( (*it).pfTauTagInfoRef().get() );
            const reco::Jet *jet = tauTagInfo->pfjetRef().get();
            m_minTheta =  100;
            m_maxTheta = -100;
            std::vector<double> phis;
            std::vector <const reco::Candidate*> candidates = jet->getJetConstituentsQuick();
            for( std::vector<const reco::Candidate*>::const_iterator candidate = candidates.begin(), candidateEnd = candidates.end();
                 candidate != candidateEnd; ++candidate )
            {
               double itheta = (*candidate)->theta();
               if( itheta > m_maxTheta ) m_maxTheta = itheta;
               if( itheta < m_minTheta ) m_minTheta = itheta;

               m_phis.push_back( (*candidate)->phi() );
            }
            if( m_minTheta > m_maxTheta ) {	
               m_minTheta = 0;
               m_maxTheta = 0;
            }

            buildBaseTau(*it, jet, comp, viewType, vc);
            m_phis.clear();
         }
         catch (std::exception&  e)
         { 
            fwLog(fwlog::kInfo) << "FWPFTauProxyBuilder missing PFTauTagInfo. Skip drawing of jets.\n";
            buildBaseTau(*it, nullptr, comp, viewType, vc);         
         }
      }
      setupAddElement( comp, product );
   }
}

REGISTER_FWPROXYBUILDER( FWPFTauProxyBuilder, reco::PFTauCollection, "PFTau", FWViewType::kAll3DBits | FWViewType::kAllRPZBits | FWViewType::kAllLegoBits);

