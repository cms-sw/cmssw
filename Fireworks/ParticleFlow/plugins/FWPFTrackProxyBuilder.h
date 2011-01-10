#ifndef _FWPFTRACKPROXYBUILDER_H_
#define _FWPFTRACKPROXYBUILDER_H_

// System include files
#include "TEveTrack.h"
#include "TEveLine.h"

// User include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Tracks/interface/estimate_field.h"

#include "DataFormats/TrackReco/interface/Track.h"

using namespace std;

class FWPFTrackProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Track>
{
   public:
      FWPFTrackProxyBuilder(){}
      virtual ~FWPFTrackProxyBuilder(){}

      REGISTER_PROXYBUILDER_METHODS();

   private:
      FWPFTrackProxyBuilder( const FWPFTrackProxyBuilder& );                     // Disable default
      const FWPFTrackProxyBuilder& operator=( const FWPFTrackProxyBuilder& );    // Disable default

      virtual bool haveSingleProduct() const { return false; } // different view types
      virtual void buildViewType(const reco::Track& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext*);
      virtual void cleanLocal();

      TEveTrack           *getTrack( unsigned int id, const reco::Track &iData );

      std::vector<TEveTrack*> tracks;
};
#endif

