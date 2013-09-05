#ifndef ThingsTSelector_h
#define ThingsTSelector_h
/** \class ThingsTSelector
 *
 * Simple interactive analysis example based on TSelector
 * accessing EDM data
 *
 * \author Luca Lista, INFN
 *
 */
#include <TH1.h>
#include "FWCore/TFWLiteSelector/interface/TFWLiteSelectorBasic.h"

namespace tfwliteselectortest {
class ThingsTSelector : public TFWLiteSelectorBasic {
public :
      ThingsTSelector() : h_a(0), h_refA(0) {}
      void begin(TList*&);
      void preProcessing(const TList*, TList&);
      void process(const edm::Event&);
      void postProcessing(TList&);
      void terminate(TList&);

private:
  /// histograms
  TH1F * h_a, *h_refA;

  ThingsTSelector(ThingsTSelector const&);
  ThingsTSelector operator=(ThingsTSelector const&);
  
  ClassDef(ThingsTSelector,2)
};
}
#endif
