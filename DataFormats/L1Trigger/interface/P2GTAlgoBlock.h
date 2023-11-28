#ifndef DataFormats_L1Trigger_P2GTAlgoBlock_h
#define DataFormats_L1Trigger_P2GTAlgoBlock_h

#include "P2GTCandidate.h"

#include <vector>
#include <string>
#include <utility>

namespace l1t {

  class P2GTAlgoBlock;
  typedef std::vector<P2GTAlgoBlock> P2GTAlgoBlockCollection;

  class P2GTAlgoBlock {
  public:
    P2GTAlgoBlock()
        : algoName_(""),
          decisionBeforeBxMaskAndPrescale_(false),
          decisionBeforePrescale_(false),
          decisionFinal_(false),
          trigObjects_() {}
    P2GTAlgoBlock(std::string name,
                  bool decisionBeforeBxMaskAndPrescale,
                  bool decisionBeforePrescale,
                  bool decisionFinal,
                  P2GTCandidateVectorRef trigObjects)
        : algoName_(std::move(name)),
          decisionBeforeBxMaskAndPrescale_(decisionBeforeBxMaskAndPrescale),
          decisionBeforePrescale_(decisionBeforePrescale),
          decisionFinal_(decisionFinal),
          trigObjects_(std::move(trigObjects)) {}

    const std::string& algoName() const { return algoName_; }
    bool decisionBeforeBxMaskAndPrescale() const { return decisionBeforeBxMaskAndPrescale_; }
    bool decisionBeforePrescale() const { return decisionBeforePrescale_; }
    bool decisionFinal() const { return decisionFinal_; }
    const P2GTCandidateVectorRef& trigObjects() const { return trigObjects_; }

  private:
    const std::string algoName_;
    const bool decisionBeforeBxMaskAndPrescale_;
    const bool decisionBeforePrescale_;
    const bool decisionFinal_;
    const P2GTCandidateVectorRef trigObjects_;
  };

}  // namespace l1t

#endif  // DataFormats_L1Trigger_P2GTAlgoBlock_h
