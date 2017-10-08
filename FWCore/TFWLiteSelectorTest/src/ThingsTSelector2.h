#ifndef ThingsTSelector2_h
#define ThingsTSelector2_h
/** \class ThingsTSelector2
 *
 * Simple interactive analysis example based on TSelector
 * accessing EDM data
 *
 * \author Luca Lista, INFN
 *
 */
#include "TH1.h"
#include "FWCore/TFWLiteSelector/interface/TFWLiteSelector.h"

namespace tfwliteselectortest {
  struct ThingsWorker {
	ThingsWorker(const TList*, TList&);
	void process( const edm::Event& iEvent );
	void postProcess(TList&);
        edm::propagate_const<TH1F*> h_a;
        edm::propagate_const<TH1F*> h_refA;
  };

  class ThingsTSelector2 : public TFWLiteSelector<ThingsWorker> {
public :
    ThingsTSelector2() {}
    void begin(TList*&) override;
    void terminate(TList&) override;
    
private:
    
    ThingsTSelector2(ThingsTSelector2 const&);
    ThingsTSelector2 operator=(ThingsTSelector2 const&);
    
    ClassDefOverride(ThingsTSelector2,2)
  };
}
#endif
