#ifndef ThingsTSelector2_h
#define ThingsTSelector2_h
/** \class ThingsTSelector2
 *
 * Simple interactive analysis example based on TSelector
 * accessing EDM data
 *
 * \author Luca Lista, INFN
 *
 * $Id: ThingsTSelector2.h,v 1.3 2006/07/07 15:56:07 chrjones Exp $
 */
#include <TH1.h>
#include "FWCore/TFWLiteSelector/interface/TFWLiteSelector.h"

namespace tfwliteselectortest {
  struct ThingsWorker {
	ThingsWorker(const TList*, TList&);
	void process( const edm::Event& iEvent );
	void postProcess(TList&);
	TH1F* h_a;
	TH1F* h_refA;
  };

  class ThingsTSelector2 : public TFWLiteSelector<ThingsWorker> {
public :
    ThingsTSelector2() {}
    void begin(TList*&);
    void terminate(TList&);
    
private:
    
    ThingsTSelector2(ThingsTSelector2 const&);
    ThingsTSelector2 operator=(ThingsTSelector2 const&);
  };
}
#endif
