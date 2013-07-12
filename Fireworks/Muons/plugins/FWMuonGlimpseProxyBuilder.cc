// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonGlimpseProxyBuilder
//
// $Id: FWMuonGlimpseProxyBuilder.cc,v 1.1 2010/07/30 08:36:01 yana Exp $
//

#include "TEveScalableStraightLineSet.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Candidates/interface/CandidateUtils.h"
#include "DataFormats/MuonReco/interface/Muon.h"

class FWMuonGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Muon>
{
public:
   FWMuonGlimpseProxyBuilder( void ) {}
   virtual ~FWMuonGlimpseProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWMuonGlimpseProxyBuilder( const FWMuonGlimpseProxyBuilder& );
   // Disable default assignment operator
   const FWMuonGlimpseProxyBuilder& operator=( const FWMuonGlimpseProxyBuilder& );

   void build( const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWMuonGlimpseProxyBuilder::build( const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* )
{
   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet( "", "" );
   marker->SetLineWidth( 2 );
   fireworks::addStraightLineSegment( marker, &iData, 1.0 );
   setupAddElement( marker, &oItemHolder );
   //add to scaler at end so that it can scale the line after all ends have been added
   // FIXME:   scaler()->addElement(marker);
}

REGISTER_FWPROXYBUILDER(FWMuonGlimpseProxyBuilder, reco::Muon, "Muons", FWViewType::kGlimpseBit);
