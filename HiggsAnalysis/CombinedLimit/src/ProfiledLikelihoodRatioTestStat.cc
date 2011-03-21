#include "HiggsAnalysis/CombinedLimit/interface/ProfiledLikelihoodRatioTestStat.h"
#include "HiggsAnalysis/CombinedLimit/interface/CloseCoutSentry.h"
#include <stdexcept>
#include <RooRealVar.h>

Double_t ProfiledLikelihoodRatioTestStat::Evaluate(RooAbsData& data, RooArgSet& nullPOI)
{
    if (data.numEntries() != 1) throw std::invalid_argument("HybridNew::TestSimpleStatistics: dataset doesn't have exactly 1 entry.");
#ifndef DEBUG
    CloseCoutSentry sentry(false);
#endif

    const RooArgSet *entry = data.get(0);
    *paramsNull_ = *entry;
    *paramsNull_ = nuisances_;
    *paramsNull_ = snapNull_;
    *paramsNull_ = nullPOI;

    if (nullPOI.getSize()) {
        RooRealVar *var = dynamic_cast<RooRealVar *>(paramsNull_->find(nullPOI.first()->GetName()));
        if (var) var->setConstant(true);
    }

#ifdef DEBUG
    paramsNull_->Print("V");
    std::cout << "Before profiling: " << pdfNull_->getVal() << std::endl;
    pdfNull_->fitTo(data, RooFit::Constrain(nuisances_), RooFit::Hesse(0));
    paramsNull_->Print("V");
    std::cout << "After profiling: " << pdfNull_->getVal() << std::endl;
#else
    pdfNull_->fitTo(data, RooFit::Constrain(nuisances_), RooFit::Hesse(0), RooFit::PrintLevel(-1), RooFit::PrintEvalErrors(-1));
#endif
    double nullNLL = pdfNull_->getVal();

    *paramsAlt_ = *entry;
    *paramsAlt_ = nuisances_;
    *paramsAlt_ = snapAlt_;

#ifdef DEBUG
    paramsNull_->Print("V");
    std::cout << "Before profiling: " << pdfAlt_->getVal() << std::endl;
    ((RooRealVar &)(*paramsNull_)[nullPOI.first()->GetName()]).setConstant(true);
    pdfAlt_->fitTo(data, RooFit::Constrain(nuisances_), RooFit::Hesse(0));
    paramsAlt_->Print("V");
    std::cout << "After profiling: " << pdfAlt_->getVal() << std::endl;
#else
    pdfAlt_->fitTo(data, RooFit::Constrain(nuisances_), RooFit::Hesse(0), RooFit::PrintLevel(-1), RooFit::PrintEvalErrors(-1));
#endif
    double altNLL = pdfAlt_->getVal();
    return -log(nullNLL/altNLL);
}

