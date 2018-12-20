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
#include "TH1.h"
#include "FWCore/TFWLiteSelector/interface/TFWLiteSelectorBasic.h"
#include "FWCore/Utilities/interface/propagate_const.h"

namespace tfwliteselectortest {
  class ThingsTSelector : public TFWLiteSelectorBasic {
  public:
    ThingsTSelector() : h_a(nullptr), h_refA(nullptr) {}
    void begin(TList*&) override;
    void preProcessing(const TList*, TList&) override;
    void process(const edm::Event&) override;
    void postProcessing(TList&) override;
    void terminate(TList&) override;

  private:
    /// histograms
    edm::propagate_const<TH1F*> h_a;
    edm::propagate_const<TH1F*> h_refA;

    ThingsTSelector(ThingsTSelector const&);
    ThingsTSelector operator=(ThingsTSelector const&);

    ClassDefOverride(ThingsTSelector, 2)
  };
}  // namespace tfwliteselectortest
#endif
