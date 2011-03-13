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
void atlas2atlas(int mass=140) {
    TFile *atlas = TFile::Open(TString::Format("atlas/atlas.mH%d.root",mass));
    if (atlas == 0) return;
    RooWorkspace *wA = (RooWorkspace *) atlas->Get("ws");
    ModelConfig  *mA = (ModelConfig  *) wA->genobj("modelConfig");
    mkNuisancePdf(wA, mA->GetNuisanceParameters());
    wA->writeToFile(TString::Format("atlas/atlas_with_nuisancePdf.mH%d.root",mass));
}
