#include "../interface/AsimovUtils.h"

#include <memory>
#include <stdexcept>
#include <TIterator.h>
#include <RooAbsData.h>
#include <RooArgSet.h>
#include <RooProdPdf.h>
#include <RooUniform.h>
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

            RooArgSet nuis(*mc->GetNuisanceParameters());
            std::auto_ptr<RooAbsPdf> nuispdf(utils::makeNuisancePdf(*mc));
            RooProdPdf *prod = dynamic_cast<RooProdPdf *>(nuispdf.get());
            if (prod == 0) throw std::runtime_error("AsimovUtils: the nuisance pdf is not a RooProdPdf!");
            std::auto_ptr<TIterator> iter(prod->pdfList().createIterator());
            for (RooAbsArg *a = (RooAbsArg *) iter->Next(); a != 0; a = (RooAbsArg *) iter->Next()) {
                RooAbsPdf *cterm = dynamic_cast<RooAbsPdf *>(a); 
                if (!cterm) throw std::logic_error("AsimovUtils: a factor of the nuisance pdf is not a Pdf!");
                if (!cterm->dependsOn(nuis)) continue; // dummy constraints
                if (typeid(*cterm) == typeid(RooUniform)) continue;
                std::auto_ptr<RooArgSet> cpars(cterm->getParameters(&gobs));
                std::auto_ptr<RooArgSet> cgobs(cterm->getObservables(&gobs));
                if (cgobs->getSize() != 1) {
                    throw std::runtime_error(Form("AsimovUtils: constraint term %s has multiple global observables", cterm->GetName()));
                }
                RooRealVar &rrv = dynamic_cast<RooRealVar &>(*cgobs->first());

                RooAbsReal *match = 0;
                if (cpars->getSize() == 1) {
                    match = dynamic_cast<RooAbsReal *>(cpars->first());
                } else {
                    std::auto_ptr<TIterator> iter2(cpars->createIterator());
                    for (RooAbsArg *a2 = (RooAbsArg *) iter2->Next(); a2 != 0; a2 = (RooAbsArg *) iter2->Next()) {
                        RooRealVar *rrv2 = dynamic_cast<RooRealVar *>(a2); 
                        if (rrv2 != 0 && !rrv2->isConstant()) {
                            if (match != 0) throw std::runtime_error(Form("AsimovUtils: constraint term %s has multiple floating params", cterm->GetName()));
                            match = rrv2;
                        }
                    }
                }
                if (match == 0) {   
                    std::cerr << "ERROR: AsimovUtils: can't find nuisance for constraint term " << cterm->GetName() << std::endl;
                    std::cerr << "Parameters: " << std::endl;
                    cpars->Print("V");
                    std::cerr << "Observables: " << std::endl;
                    cgobs->Print("V");
                    throw std::runtime_error(Form("AsimovUtils: can't find nuisance for constraint term %s", cterm->GetName()));
                }
                rrv.setVal(match->getVal());
            }

            // snapshot
            snapshot.removeAll();
            utils::setAllConstant(gobs, true);
            gobs.snapshot(snapshot);

            // revert things to normal
            gobs = snapGlobalObsData;
    
            if (verbose > 1) {
                std::cout << "Global observables for data: " << std::endl;
                snapGlobalObsData.Print("V");
                std::cout << "Global observables for asimov: " << std::endl;
                snapshot.Print("V");
            }
        }

        return asimov;
}
