using namespace RooStats;
using namespace RooFit;

void importSet(TString setName, RooWorkspace *comb, const RooArgSet *atlas, const RooArgSet *cms) {
    RooArgSet set;
    std::cout << "Will try to import " << setName << "." << std::endl;
    TIterator *it = atlas->createIterator();
    RooAbsArg *arg;
    while ((arg = (RooAbsArg *) it->Next()) != 0) {
        TString name = arg->GetName();
        std::cout << " [from atlas] " << name << std::endl;
        if (name == "LUMI" || name == "XS_GG" || name == "LUMI_mean" || name == "XS_GG_mean") continue;
        RooAbsArg *renamed = comb->var(name+"_atlas");
        if (renamed == 0) { std::cerr << "Error: " << name << " of type " << arg->ClassName() << " was not imported from atlas." << std::endl; continue; }
        set.add(*renamed);
    }
    delete it;
    it = cms->createIterator();
    while ((arg = (RooAbsArg *) it->Next()) != 0) {
        TString name = arg->GetName();
        std::cout << " [from cms  ] " << name << std::endl;
        if (name == "theta_Lumi" || name == "theta_Higgs_XS" || name == "thetaIn_Lumi" || name == "thetaIn_Higgs_XS") {
            RooAbsArg *renamed = comb->var(name);
            if (renamed == 0) { std::cerr << "Error: " << name << " of type " << arg->ClassName() << " was not imported from cms." << std::endl; continue; }
            set.add(*renamed);
        } else {
            RooAbsArg *renamed = comb->var(name+"_cms");
            if (renamed == 0) { std::cerr << "Error: " << name << " of type " << arg->ClassName() << " was not imported from cms." << std::endl; continue; }
            set.add(*renamed);
        }
    }
    delete it;
    comb->defineSet(setName, set);
    std::cout << "Imported and merged " << setName << ":" << std::endl;
    set.Print("V");
}

void importData(TString dataName, RooWorkspace *comb, const RooArgSet *atlas, const RooArgSet *cms) {
    RooArgSet obs(*comb->set("observables"));

    TIterator *it = atlas->createIterator();
    RooRealVar *arg;
    while ((arg = (RooRealVar *) it->Next()) != 0) {
        TString name = arg->GetName();
        std::cout << " [from atlas] " << name << " = " << arg->getVal() << std::endl;
        obs.setRealValue(name+"_atlas", arg->getVal());
    }
    delete it;
    it = cms->createIterator();
    while ((arg = (RooRealVar *) it->Next()) != 0) {
        TString name = arg->GetName();
        std::cout << " [from cms  ] " << name << " = " << arg->getVal() << std::endl;
        obs.setRealValue(name+"_cms", arg->getVal());
    }
    delete it;

    RooDataSet *data_obs = new RooDataSet(dataName, "Combined Data", obs);
    data_obs->add(obs);
    comb->import(*data_obs);
    std::cout << "Imported and merged DataSet "<< dataName<<":" << std::endl;
    obs.Print("V");
}


void mkCombModel(int mass=140) {
    TFile *atlas = TFile::Open(TString::Format("atlas/atlas.mH%d.root",mass));
    TFile *cms   = TFile::Open(TString::Format("cms/higgsCombineHWW.ProfileLikelihood.mH%d.root",mass));
    if (atlas == 0 || cms == 0) return;
    RooWorkspace *wA = (RooWorkspace *) atlas->Get("ws");
    RooWorkspace *wC = (RooWorkspace *) cms->Get("w");
    ModelConfig  *mA = (ModelConfig  *) wA->genobj("modelConfig");
    ModelConfig  *mC = (ModelConfig  *) wC->genobj("ModelConfig");
    std::cout << "Atlas model:\n"; mA->Print("V");
    std::cout << "CMS model:\n";   mC->Print("V");
    RooWorkspace *w = new RooWorkspace("w","w");
    w->import(*mA->GetPdf(), RenameAllNodes("atlas"), RenameAllVariablesExcept("atlas","mu"));
    w->import(*mC->GetPdf(), RenameAllNodes("cms"),   RenameAllVariablesExcept("cms","r,theta_Lumi,thetaIn_Lumi,theta_Higgs_XS,thetaIn_Higgs_XS"));
    w->Print("V");

    w->defineSet("POI","r");
    importSet("observables",       w, mA->GetObservables(),        mC->GetObservables());
    importSet("nuisances",         w, mA->GetNuisanceParameters(), mC->GetNuisanceParameters());
    importSet("globalObservables", w, mA->GetGlobalObservables(),  mC->GetGlobalObservables());
    importData("data_obs", w, wA->data("asimovData_0")->get(0), wC->data("data_obs")->get(0));

    w->factory(TString::Format("PROD::_naive_model_s(%s_atlas,%s_cms)", mA->GetPdf()->GetName(), mC->GetPdf()->GetName()));
    w->factory("Uniform::_dummy_pdf(theta_Lumi)");

    RooCustomizer make_model_s(*w->pdf("_naive_model_s"), "model_s");
    make_model_s.replaceArg(*w->var("mu"),    *w->var("r"));
    make_model_s.replaceArg(*w->pdf("LUMI_gaus_atlas"), *w->pdf("_dummy_pdf"));
    make_model_s.replaceArg(*w->var("XS_GG_atlas"), *w->var("theta_Higgs_XS"));
    make_model_s.replaceArg(*w->var("LUMI_atlas"),  *w->var("theta_Lumi"));
    make_model_s.replaceArg(*w->pdf("XS_GG_gaus_atlas"), *w->pdf("_dummy_pdf"));
    RooAbsPdf *model_s = make_model_s.build(); model_s->SetName("model_s");
    w->import(*model_s);
    std::cout << "Created top-level pdf model_s. Parameters: " << std::endl;
    model_s->getParameters(w->data("data_obs"))->Print("V");

    RooConstVar *zorro = new RooConstVar("__zero__","Zero", 0); w->import(*zorro);
    w->factory("EDIT::model_b(model_s,r=__zero__)");

    ModelConfig *m = new ModelConfig("ModelConfig", w);
    m->SetPdf(*model_s);
    m->SetParametersOfInterest(*w->set("POI"));
    m->SetNuisanceParameters(*w->set("nuisances"));
    m->SetGlobalObservables(*w->set("globalObservables"));
    m->SetObservables(*w->set("observables"));
    w->import(*m, "ModelConfig");

    w->writeToFile(TString::Format("comb/comb.mH%d.root",mass));
}
