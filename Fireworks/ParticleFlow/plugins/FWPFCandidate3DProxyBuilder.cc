// -*- C++ -*-
//
// Package:     ParticleFlow
// Class  :     FWCandidate3DProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Colin Bernet
//         Created:  Fri May 28 15:58:19 CEST 2010
// Edited:           sharris, Wed 9 Feb 2011, 17:34
//

// System include files
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TRandom3.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

// User include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/ParticleFlow/interface/setTrackTypePF.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "TEveCompound.h"
#include "TEveBoxSet.h"

//-----------------------------------------------------------------------------
// FWPFCandidate3DProxyBuilder
//-----------------------------------------------------------------------------

class FWPFCandidate3DProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFCandidate>
{
      
   public:
   // ---------------- Constructor(s)/Destructor ----------------------
  FWPFCandidate3DProxyBuilder() { myRandom.SetSeed(0); }
      virtual ~FWPFCandidate3DProxyBuilder();
   
      REGISTER_PROXYBUILDER_METHODS();

   private:
  TRandom3 myRandom;

      FWPFCandidate3DProxyBuilder( const FWPFCandidate3DProxyBuilder& );                    // Stop default
      const FWPFCandidate3DProxyBuilder& operator=( const FWPFCandidate3DProxyBuilder& );   // Stop default

   // --------------------- Member Functions --------------------------
      void build( const reco::PFCandidate& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );

};
//=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_



//______________________________________________________________________________
FWPFCandidate3DProxyBuilder::~FWPFCandidate3DProxyBuilder(){}

//______________________________________________________________________________
void 
FWPFCandidate3DProxyBuilder::build( const reco::PFCandidate& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) 
{
  TEveCompound* comp = createCompound(false,true);  
  
  const reco::PFCandidate::ElementsInBlocks& elems = iData.elementsInBlocks();
  
  for( unsigned i = 0 ; i < elems.size(); ++i ) {
    const reco::PFBlockElement& elem = elems[i].first->elements()[elems[i].second];
    switch( elem.type() ) {
    case reco::PFBlockElement::TRACK:
      {
	TEveRecTrack t;
	t.fBeta = 1.;
	t.fP = TEveVector( iData.px(), iData.py(), iData.pz() );
	t.fV = TEveVector( iData.vertex().x(), iData.vertex().y(), iData.vertex().z() );
	t.fSign = iData.charge();
	TEveTrack* trk = new TEveTrack(&t, context().getTrackPropagator() );      
	trk->MakeTrack();      
	fireworks::setTrackTypePF( iData, trk );    
	setupAddElement( trk, comp, false );
      }
      break;
    case reco::PFBlockElement::ECAL:
    case reco::PFBlockElement::HCAL:
    case reco::PFBlockElement::HGC_ECAL:
    case reco::PFBlockElement::HGC_HCALF:
    case reco::PFBlockElement::HGC_HCALB:
      {
	if( elem.clusterRef().isNull() || !elem.clusterRef().isAvailable() ) {
	  TEveRecTrack t;
	  t.fBeta = 1.;
	  t.fP = TEveVector( iData.px(), iData.py(), iData.pz() );
	  t.fV = TEveVector( iData.vertex().x(), iData.vertex().y(), iData.vertex().z() );
	  t.fSign = iData.charge();
	  TEveTrack* trk = new TEveTrack(&t, context().getTrackPropagator() );      
	  trk->MakeTrack();      
	  fireworks::setTrackTypePF( iData, trk );    
	  setupAddElement( trk, comp, false );
	  continue;
	}
	const std::vector<std::pair<DetId, float> >& clusterDetIds = 
	  elem.clusterRef()->hitsAndFractions();	
	TEveBoxSet* boxset = new TEveBoxSet();
	boxset->Reset(TEveBoxSet::kBT_FreeBox, true, clusterDetIds.size());
	boxset->UseSingleColor();
	boxset->SetPickable(1);
	//const unsigned color = (unsigned)myRandom.Uniform(50);	
	for( std::vector<std::pair<DetId, float> >::const_iterator it = clusterDetIds.begin(), itEnd = clusterDetIds.end();
	     it != itEnd; ++it )
	  {
	    const float* corners = item()->getGeom()->getCorners( (*it).first );
	    if( corners == 0 ) {
	      continue;
	    }
	    std::vector<float> pnts(24);    
	    fireworks::energyTower3DCorners(corners, (*it).second, pnts);
	    boxset->AddBox( &pnts[0]);
	    //boxset->DigitColor( color + 50, 50);     
	  }
	
	boxset->RefitPlex();
	setupAddElement(boxset,comp,false);
      }
      break;
    default:
      break;
    }
  }
  
  comp->SetMainColor((unsigned)2.0*myRandom.Uniform(50));

  setupAddElement( comp, &oItemHolder, false );
}

//______________________________________________________________________________
REGISTER_FWPROXYBUILDER(FWPFCandidate3DProxyBuilder, reco::PFCandidate,"PF Candidates", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
