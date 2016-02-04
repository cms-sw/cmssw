#ifndef _FWPFPATJET3DPROXYBUILDER__
#define _FWPFPATJET3DPROXYBUILDER__

// User include files

#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveVSDStructs.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/ParticleFlow/interface/setTrackTypePF.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/PatCandidates/interface/Jet.h"   /* Included for pat::Jet declaration */



 /***************************************************\
(            FORWARD DECLARATIONS                     )
 \***************************************************/

template<class T>
class FWPFPatJet3DProxyBuilder : public FWSimpleProxyBuilderTemplate<T> {

public:
   FWPFPatJet3DProxyBuilder();
   virtual ~FWPFPatJet3DProxyBuilder();


 /***************************************************\
(            MEMBER FUNCTIONS                         )
 \***************************************************/

private:
   FWPFPatJet3DProxyBuilder(const FWPFPatJet3DProxyBuilder&); // Stop default
   const FWPFPatJet3DProxyBuilder& operator=(const FWPFPatJet3DProxyBuilder&); // Stop default

   void build(const T&, unsigned int, TEveElement&, const FWViewContext*);


 /***************************************************\
(            MEMBER DATA                              )
 \***************************************************/
};
#endif
