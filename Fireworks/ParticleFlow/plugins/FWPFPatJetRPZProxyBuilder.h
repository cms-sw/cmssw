#ifndef _FWPFPATJETRPZPROXYBUILDER_H_
#define _FWPFPATJETRPZPROXYBUILDER_H_

// User include files

#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveVSDStructs.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/ParticleFlow/interface/setTrackTypePF.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

 /*************************************************************************\
(				FORWARD DECLARATIONS	       		    )
 \*************************************************************************/

template<class T>
class FWPFPatJetRPZProxyBuilder : public FWSimpleProxyBuilderTemplate<T> {
public:
	FWPFPatJetRPZProxyBuilder();
	virtual ~FWPFPatJetRPZProxyBuilder();

 /*************************************************************************\
(				MEMBER FUNCTIONS	       		    )
 \*************************************************************************/

private:
	FWPFPatJetRPZProxyBuilder(const FWPFPatJetRPZProxyBuilder&);			//stop default
	const FWPFPatJetRPZProxyBuilder& operator=(const FWPFPatJetRPZProxyBuilder&);	//stop default

	void build(const T&, unsigned int, TEveElement&, const FWViewContext*);

};
#endif
