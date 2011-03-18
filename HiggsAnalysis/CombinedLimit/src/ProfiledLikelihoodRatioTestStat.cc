#include "HiggsAnalysis/CombinedLimit/interface/ProfiledLikelihoodRatioTestStat.h"
#include "HiggsAnalysis/CombinedLimit/interface/CloseCoutSentry.h"
#include <stdexcept>
#include <RooRealVar.h>

Double_t ProfiledLikelihoodRatioTestStat::Evaluate(RooAbsData& data, RooArgSet& nullPOI)
{
    if (data.numEntries() != 1) throw std::invalid_argument("HybridNew::TestSimpleStatistics: dataset doesn't have exactly 1 entry.");
    CloseCoutSentry sentry(true);
    const RooArgSet *entry = data.get(0);
    *paramsNull_ = *entry;
    *paramsNull_ = nuisances_;
    *paramsNull_ = snapNull_;
    *paramsNull_ = nullPOI;
    ((RooRealVar &)(*paramsNull_)[nullPOI.first()->GetName()]).setConstant(true);
    pdfNull_->fitTo(data, RooFit::Constrain(nuisances_), RooFit::Hesse(0), RooFit::PrintLevel(-1), RooFit::PrintEvalErrors(-1));
    double nullNLL = pdfNull_->getVal();

    *paramsAlt_ = *entry;
    *paramsAlt_ = nuisances_;
    *paramsAlt_ = snapAlt_;
    ((RooRealVar &)(*paramsNull_)[nullPOI.first()->GetName()]).setConstant(true);
    pdfAlt_->fitTo(data, RooFit::Constrain(nuisances_), RooFit::Hesse(0), RooFit::PrintLevel(-1), RooFit::PrintEvalErrors(-1));
    double altNLL = pdfAlt_->getVal();

    return -log(nullNLL/altNLL);
}

