#include <TH1.h>
#include <TFile.h>
#include <cmath>
#include <iostream>
#include <string>
#include <map>

TH1 * EA_mu = 0;
TH1 * EA_el = 0;


bool loadEAHisto(const std::string &histoName, const char *file, const char *name) {
    TH1 **histo = 0;
    if (histoName == "EA_mu") histo = & EA_mu;
    if (histoName == "EA_el") histo = & EA_el;
    if (histo == 0)  {
        std::cerr << "ERROR: histogram " << histoName << " is not defined in fakeRate.cc." << std::endl;
        return 0;
    }

    if (*histo != 0) delete *histo;
    TFile *f = TFile::Open(file);
    if (f->Get(name) == 0) {
        std::cerr << "ERROR: could not find " << name << " in " << file << std::endl;
        *histo = 0;
    } else {
        *histo = (TH1*) f->Get(name)->Clone(name);
        (*histo)->SetDirectory(0);
    }
    f->Close();
    return histo != 0;
}

float eaCorr(float neutralIso, int pdgId, float eta, float rho) 
{
    TH1 *hist = (abs(pdgId)==11 ? EA_el : EA_mu);
    int etabin = std::max(1, std::min(hist->GetNbinsX(), hist->GetXaxis()->FindBin(std::abs(eta))));
    return std::max<float>(neutralIso - rho * hist->GetBinContent(etabin), 0.);
}

void rhoCorrections() {}
