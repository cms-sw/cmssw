using namespace RooStats;
using namespace RooFit;

void mkNuisancePdf(RooWorkspace *w, const RooArgSet *nuis) {
    RooArgList pdfs;
    std::cout << "Will try to collect pdf for nuisances:" << std::endl;
    TIterator *it = nuis->createIterator();
    RooAbsArg *arg; 
    while ((arg = (RooAbsArg *) it->Next()) != 0) {
        TString name = arg->GetName(), tryname;
        std::cout << " - " << name << ": ";
        RooAbsPdf *pdf = 0;
        TString tryname_gaus  = name+"_gaus";
        TString tryname_Gamma = name+"_Gamma";
        bool hasGaus  = (w->pdf(tryname_gaus)  != 0);
        bool hasGamma = (w->pdf(tryname_Gamma) != 0);
        if (hasGaus && hasGamma) { 
            std::cout << " ERROR: both " << tryname_gaus << " and " << tryname_Gamma << " exits!" << std::endl; 
            continue; 
        } else if (hasGaus) {
            pdf = w->pdf(tryname_gaus);
        } else if (hasGamma) {
            pdf = w->pdf(tryname_Gamma);
        }
        std::cout << (pdf == 0 ? "NOT FOUND!" : pdf->ClassName()) << std::endl;
        if (pdf) pdfs.add(*pdf);
    }
    RooProdPdf *prod = new RooProdPdf("nuisancePdf","nuisancePdf",pdfs);
    w->import(*prod);
    delete it;
}

void collectObsFactors(RooProdPdf *pdf, const RooArgSet &observables, RooArgList &terms) {
    RooArgList list(pdf->pdfList());
    std::cout << "pdf " << pdf->GetName() << " is a product of " << list.getSize() << " terms." << std::endl;
    for (int i = 0, n = list.getSize(); i < n; ++i) {
        RooAbsPdf *pdfi = (RooAbsPdf *) list.at(i);
        std::cout << "   pdf " << pdfi->GetName() << " (" << pdfi->ClassName() << ") ";
        if (pdfi->ClassName() == std::string("RooProdPdf")) {
            std::cout << " is product: iterate again." << std::endl;
            collectObsFactors((RooProdPdf*)pdfi, observables, terms);
        } else if (pdfi->dependsOn(observables)) {
            terms.add(*pdfi);
            std::cout << " depends on observables: included." << std::endl;
        } else {
            std::cout << " does not depend on observables: ignored." << std::endl;
        }
    }
}

void mkModelObsPdf(RooWorkspace *w, RooAbsPdf *model_s, const RooArgSet &observables) {
    if (model_s->ClassName() != std::string("RooProdPdf")) {
        std::cout << "Error: "<<model_s->GetName()<<" is not a RooProdPdf. Can't optimize." << std::endl;
        return;
    } 
    RooArgList factors;
    collectObsFactors((RooProdPdf*)model_s, observables, factors);
    RooProdPdf *modelObs_s = new RooProdPdf("modelObs_s", "Part of model_s that depends on observables", factors);
    w->import(*modelObs_s);
    std::cout << "Created " << modelObs_s->GetName() << " from product of " << std::endl;
    factors.Print("V");
}

void atlas2atlas(int mass=140) {
    TFile *atlas = TFile::Open(TString::Format("atlas/atlas.mH%d.root",mass));
    if (atlas == 0) return;
    RooWorkspace *wA = (RooWorkspace *) atlas->Get("ws");
    ModelConfig  *mA = (ModelConfig  *) wA->genobj("modelConfig");
    mkNuisancePdf(wA, mA->GetNuisanceParameters());
    // does not work
    //mkModelObsPdf(wA, mA->GetPdf(), *mA->GetObservables());
    //wA->import(*(new RooConstVar("__modelObs_b_zero_","Yet another zero",0.0)));
    //wA->factory("EDIT::modelObs_b(modelObs_s, mu=__modelObs_b_zero_)");
    wA->writeToFile(TString::Format("atlas/atlas_with_nuisancePdf.mH%d.root",mass));
}
