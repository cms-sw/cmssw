#ifndef HiggsAnalysis_CombinedLimit_RooFitGlobalKillSentry_
#define HiggsAnalysis_CombinedLimit_RooFitGlobalKillSentry_

#include <RooMsgService.h>

/** This class sets the global kill for RooFit messages to some value when created,
    and resets it to the old value when destroyed. */
class RooFitGlobalKillSentry {
    public:
        RooFitGlobalKillSentry(RooFit::MsgLevel level = RooFit::WARNING) :
            globalKill_(RooMsgService::instance().globalKillBelow())
    {
        RooMsgService::instance().setGlobalKillBelow(level);
    }

        ~RooFitGlobalKillSentry() {
            RooMsgService::instance().setGlobalKillBelow(globalKill_);
        }
    private:
        RooFit::MsgLevel globalKill_;
};
#endif
