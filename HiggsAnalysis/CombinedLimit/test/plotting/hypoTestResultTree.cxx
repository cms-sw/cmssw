
void hypoTestResultTree(TString fOutName, double mass, double rValue=1.0, const char *poiName="r") {
    if (gROOT->GetListOfFiles()->GetSize() == 0) {
        std::cerr << "ERROR: you have to open at least one root file" << std::endl;
    }
    TFile *fOut = new TFile(fOutName, "RECREATE");
    TTree *tree = new TTree("q","Test statistics");
    float q, mh = mass, r = rValue, weight; int type;
    tree->Branch("q", &q, "q/F");
    tree->Branch("mh", &mh, "mh/F");
    tree->Branch("weight", &weight, "weight/F");
    tree->Branch("type", &type, "type/I");
    tree->Branch("r", &r, "r/F");
    TString prefix1 = TString::Format("HypoTestResult_mh%g_%s%g_",mass,poiName,rValue);
    TString prefix2 = TString::Format("HypoTestResult_%s%g_",poiName,rValue);
    long int nS = 0, nB = 0;
    for (int i = 0, n = gROOT->GetListOfFiles()->GetSize()-1;  i < n; ++i) {
        TDirectory *toyDir = ((TFile*) gROOT->GetListOfFiles()->At(i))->GetDirectory("toys");
        if (toyDir == 0) {
            std::cerr << "Error in file " << gROOT->GetListOfFiles()->At(i)->GetName() << ": directory /toys not found" << std::endl;
            continue;
        }
        TIter next(toyDir->GetListOfKeys()); TKey *k;
        while ((k = (TKey *) next()) != 0) {
            if (TString(k->GetName()).Index(prefix1) != 0 && TString(k->GetName()).Index(prefix2) != 0) continue;
            RooStats::HypoTestResult *toy = dynamic_cast<RooStats::HypoTestResult *>(toyDir->Get(k->GetName()));
            if (toy == 0) continue;
            std::cout << " - " << k->GetName() << std::endl;
            RooStats::SamplingDistribution * bDistribution = toy->GetNullDistribution(), * sDistribution = toy->GetAltDistribution();
            const std::vector<Double_t> & bdist   = bDistribution->GetSamplingDistribution();
            const std::vector<Double_t> & bweight = bDistribution->GetSampleWeights();
            for (int j = 0, nj = bdist.size(); j < nj; ++j) {
                q = bdist[j]; weight = bweight[j]; type = -1;
                tree->Fill(); nB++;
            }

            const std::vector<Double_t> & sdist   = sDistribution->GetSamplingDistribution();
            const std::vector<Double_t> & sweight = sDistribution->GetSampleWeights();
            for (int j = 0, nj = sdist.size(); j < nj; ++j) {
                q = sdist[j]; weight = sweight[j]; type = 1;
                tree->Fill(); nS++;
            }

            Double_t data =  toy->GetTestStatisticData();
            weight = 1.0; q = data; type = 0;
            tree->Fill();
        }
    }
    tree->Write();
    fOut->Close();
    std::cout << "Saved test statistics distributions for " << nS << " signal toys and " << nB << " background toys to " << fOutName << "." << std::endl;
}
