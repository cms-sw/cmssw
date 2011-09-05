#include "../interface/AsimovUtils.h"

#include <memory>
#include <TIterator.h>
#include <RooAbsData.h>
#include <RooArgSet.h>
#include "../interface/utils.h"
#include "../interface/ToyMCSamplerOpt.h"
#include "../interface/CloseCoutSentry.h"

RooAbsData *asimovutils::asimovDatasetNominal(RooStats::ModelConfig *mc, double poiValue, int verbose) {
        RooArgSet  poi(*mc->GetParametersOfInterest());
        RooRealVar *r = dynamic_cast<RooRealVar *>(poi.first());
        r->setConstant(true); r->setVal(poiValue);
        toymcoptutils::SimPdfGenInfo newToyMC(*mc->GetPdf(), *mc->GetObservables(), false); 
        RooRealVar *weightVar = 0;
        RooAbsData *asimov = newToyMC.generateAsimov(weightVar); 
        delete weightVar;
        return asimov;
}

RooAbsData *asimovutils::asimovDatasetWithFit(RooStats::ModelConfig *mc, RooAbsData &realdata, RooAbsCollection &snapshot, double poiValue, int verbose) {
        RooArgSet  poi(*mc->GetParametersOfInterest());
        RooRealVar *r = dynamic_cast<RooRealVar *>(poi.first());
        r->setConstant(true); r->setVal(poiValue);
        {
            CloseCoutSentry sentry(verbose < 3);
            if (mc->GetNuisanceParameters()) {
                mc->GetPdf()->fitTo(realdata, RooFit::Minimizer("Minuit2","minimize"), RooFit::Strategy(1), RooFit::Constrain(*mc->GetNuisanceParameters()));
            } else {
                // Do we have free parameters anyway that need fitting?
                bool hasFloatParams = false;
                std::auto_ptr<RooArgSet> params(mc->GetPdf()->getParameters(realdata));
                std::auto_ptr<TIterator> iter(params->createIterator());
                for (RooAbsArg *a = (RooAbsArg *) iter->Next(); a != 0; a = (RooAbsArg *) iter->Next()) {
                    RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
                    if ( rrv != 0 && rrv->isConstant() == false ) { hasFloatParams = true; break; }
                } 
                if (hasFloatParams) mc->GetPdf()->fitTo(realdata, RooFit::Minimizer("Minuit2","minimize"), RooFit::Strategy(1));
            }
        }
        if (mc->GetNuisanceParameters() && verbose > 1) {
            std::cout << "Nuisance parameters after fit for asimov dataset: " << std::endl;
            mc->GetNuisanceParameters()->Print("V");
        }
        toymcoptutils::SimPdfGenInfo newToyMC(*mc->GetPdf(), *mc->GetObservables(), false); 
        RooRealVar *weightVar = 0;
        RooAbsData *asimov = newToyMC.generateAsimov(weightVar); 
        delete weightVar;

        // NOW SNAPSHOT THE GLOBAL OBSERVABLES
        if (mc->GetGlobalObservables() && mc->GetGlobalObservables()->getSize() > 0) {
            RooArgSet gobs(*mc->GetGlobalObservables());

            // snapshot data global observables
            RooArgSet snapGlobalObsData;
            utils::setAllConstant(gobs, true);
            gobs.snapshot(snapGlobalObsData);

            // now get the ones for the asimov dataset.
            // we do this by fitting the nuisance pdf with floating global observables but fixed nuisances (which we call observables)
            // part 1: create the nuisance pdf
            std::auto_ptr<RooAbsPdf> nuispdf(utils::makeNuisancePdf(*mc));
            // part 2: create the dataset containing the nuisances
            RooArgSet nuis(*mc->GetNuisanceParameters());
            RooDataSet nuisdata("nuisData","nuisData", nuis);
            nuisdata.add(nuis);
            // part 3: make everything constant except the global observables which have to be floating
            //         remember what done, to be able to undo it afterwards
            RooArgSet paramsSetToConstants;
            std::auto_ptr<RooArgSet> nuispdfparams(nuispdf->getParameters(nuisdata)); 
            std::auto_ptr<TIterator> iter(nuispdfparams->createIterator());
            for (RooAbsArg *a = (RooAbsArg *) iter->Next(); a != 0; a = (RooAbsArg *) iter->Next()) {
                RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
                if (rrv) {
                    if (gobs.find(rrv->GetName())) {
                        rrv->setConstant(false);
                    } else {
                        if (!rrv->isConstant()) paramsSetToConstants.add(*rrv);
                        rrv->setConstant(true);
                    }
                }
            }
            {
                CloseCoutSentry sentry(verbose < 3);
                nuispdf->fitTo(nuisdata, RooFit::Minimizer("Minuit2","minimize"), RooFit::Strategy(1));
            }
            // snapshot
            snapshot.removeAll();
            utils::setAllConstant(gobs, true);
            gobs.snapshot(snapshot);

            // revert things to normal
            gobs = snapGlobalObsData;
            utils::setAllConstant(paramsSetToConstants, false);
        }

        return asimov;
}
