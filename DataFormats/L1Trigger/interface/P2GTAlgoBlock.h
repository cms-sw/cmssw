#ifndef DataFormats_L1Trigger_P2GTAlgoBlock_h
#define DataFormats_L1Trigger_P2GTAlgoBlock_h

#include "P2GTCandidate.h"

#include <map>
#include <string>

namespace l1t {

  class P2GTAlgoBlock;
  typedef std::map<std::string, P2GTAlgoBlock> P2GTAlgoBlockMap;

  class P2GTAlgoBlock {
  public:
    P2GTAlgoBlock()
        : decisionBeforeBxMaskAndPrescale_(false),
          decisionBeforePrescale_(false),
          decisionFinal_(false),
          decisionFinalPreview_(false),
          isVeto_(false),
          triggerTypes_(0),
          trigObjects_() {}

    P2GTAlgoBlock(bool decisionBeforeBxMaskAndPrescale,
                  bool decisionBeforePrescale,
                  bool decisionFinal,
                  bool decisionFinalPreview,
                  bool isVeto,
                  int triggerTypes,
                  P2GTCandidateVectorRef trigObjects)
        : decisionBeforeBxMaskAndPrescale_(decisionBeforeBxMaskAndPrescale),
          decisionBeforePrescale_(decisionBeforePrescale),
          decisionFinal_(decisionFinal),
          decisionFinalPreview_(decisionFinalPreview),
          isVeto_(isVeto),
          triggerTypes_(triggerTypes),
          trigObjects_(std::move(trigObjects)) {}

    bool decisionBeforeBxMaskAndPrescale() const { return decisionBeforeBxMaskAndPrescale_; }
    bool decisionBeforePrescale() const { return decisionBeforePrescale_; }
    bool decisionFinal() const { return decisionFinal_; }
    bool decisionFinalPreview() const { return decisionFinalPreview_; }
    bool isVeto() const { return isVeto_; }
    int triggerTypes() const { return triggerTypes_; }
    const P2GTCandidateVectorRef& trigObjects() const { return trigObjects_; }

  private:
    const bool decisionBeforeBxMaskAndPrescale_;
    const bool decisionBeforePrescale_;
    const bool decisionFinal_;
    const bool decisionFinalPreview_;
    const bool isVeto_;
    const int triggerTypes_;
    const P2GTCandidateVectorRef trigObjects_;
  };

}  // namespace l1t

#endif  // DataFormats_L1Trigger_P2GTAlgoBlock_h
