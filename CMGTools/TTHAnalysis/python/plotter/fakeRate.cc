#include <TH2.h>
#include <TFile.h>
#include <cmath>
#include <iostream>
#include <string>
#include <map>

TH2 * FR_mu = 0;
TH2 * FR2_mu = 0;
TH2 * FR3_mu = 0;
TH2 * FR4_mu = 0;
TH2 * FR5_mu = 0;
TH2 * FR_el = 0;
TH2 * FR2_el = 0;
TH2 * FR3_el = 0;
TH2 * FR4_el = 0;
TH2 * FR5_el = 0;
TH2 * QF_el = 0;
TH2 * FRi_mu[6], *FRi_el[6];


bool loadFRHisto(const std::string &histoName, const char *file, const char *name) {
    TH2 **histo = 0, **hptr2 = 0;
    if      (histoName == "FR_mu")  { histo = & FR_mu;  hptr2 = & FRi_mu[0]; }
    else if (histoName == "FR_el")  { histo = & FR_el;  hptr2 = & FRi_el[0]; }
    else if (histoName == "FR2_mu") { histo = & FR2_mu; hptr2 = & FRi_mu[2]; }
    else if (histoName == "FR2_el") { histo = & FR2_el; hptr2 = & FRi_el[2]; }
    else if (histoName == "FR3_mu") { histo = & FR3_mu; hptr2 = & FRi_mu[3]; }
    else if (histoName == "FR3_el") { histo = & FR3_el; hptr2 = & FRi_el[3]; }
    else if (histoName == "FR4_mu") { histo = & FR4_mu; hptr2 = & FRi_mu[4]; }
    else if (histoName == "FR4_el") { histo = & FR4_el; hptr2 = & FRi_el[4]; }
    else if (histoName == "FR5_mu") { histo = & FR5_mu; hptr2 = & FRi_mu[5]; }
    else if (histoName == "FR5_el") { histo = & FR5_el; hptr2 = & FRi_el[5]; }
    else if (histoName == "QF_el") histo = & QF_el;
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
        *histo = (TH2*) f->Get(name)->Clone(name);
        (*histo)->SetDirectory(0);
        if (hptr2) *hptr2 = *histo;
    }
    f->Close();
    return histo != 0;
}

float fakeRateWeight_2lss(float l1pt, float l1eta, int l1pdgId, float l1mva,
                         float l2pt, float l2eta, int l2pdgId, float l2mva, float WP) 
{
    int nfail = (l1mva < WP)+(l2mva < WP);
    switch (nfail) {
        case 1: {
            double fpt,feta; int fid;
            if (l1mva < l2mva) { fpt = l1pt; feta = std::abs(l1eta); fid = abs(l1pdgId); }
            else               { fpt = l2pt; feta = std::abs(l2eta); fid = abs(l2pdgId); }
            TH2 *hist = (fid == 11 ? FR_el : FR_mu);
            int ptbin  = std::max(1, std::min(hist->GetNbinsX(), hist->GetXaxis()->FindBin(fpt)));
            int etabin = std::max(1, std::min(hist->GetNbinsY(), hist->GetYaxis()->FindBin(feta)));
            double fr = hist->GetBinContent(ptbin,etabin);
            return fr/(1-fr);
        }
        case 2: {
            TH2 *hist1 = (abs(l1pdgId) == 11 ? FR_el : FR_mu);
            int ptbin1  = std::max(1, std::min(hist1->GetNbinsX(), hist1->GetXaxis()->FindBin(l1pt)));
            int etabin1 = std::max(1, std::min(hist1->GetNbinsY(), hist1->GetYaxis()->FindBin(std::abs(l1eta))));
            double fr1 = hist1->GetBinContent(ptbin1,etabin1);
            TH2 *hist2 = (abs(l2pdgId) == 11 ? FR_el : FR_mu);
            int ptbin2  = std::max(1, std::min(hist2->GetNbinsX(), hist2->GetXaxis()->FindBin(l2pt)));
            int etabin2 = std::max(1, std::min(hist2->GetNbinsY(), hist2->GetYaxis()->FindBin(std::abs(l2eta))));
            double fr2 = hist2->GetBinContent(ptbin2,etabin2);
            return -fr1*fr2/((1-fr1)*(1-fr2));
        }
        default: return 0;
    }
}



float fakeRateWeight_2lssCB_i(float l1pt, float l1eta, int l1pdgId, float l1relIso,
                            float l2pt, float l2eta, int l2pdgId, float l2relIso, float WP, int iFR) 
{
    int nfail = (l1relIso > WP)+(l2relIso > WP);
    switch (nfail) {
        case 1: {
            double fpt,feta; int fid;
            if (l1relIso > l2relIso) { fpt = l1pt; feta = std::abs(l1eta); fid = abs(l1pdgId); }
            else                     { fpt = l2pt; feta = std::abs(l2eta); fid = abs(l2pdgId); }
            TH2 *hist = (fid == 11 ? FRi_el[iFR] : FRi_mu[iFR]);
            int ptbin  = std::max(1, std::min(hist->GetNbinsX(), hist->GetXaxis()->FindBin(fpt)));
            int etabin = std::max(1, std::min(hist->GetNbinsY(), hist->GetYaxis()->FindBin(feta)));
            double fr = hist->GetBinContent(ptbin,etabin);
            return fr/(1-fr);
        }
        case 2: {
            TH2 *hist1 = (abs(l1pdgId) == 11 ? FRi_el[iFR] : FRi_mu[iFR]);
            int ptbin1  = std::max(1, std::min(hist1->GetNbinsX(), hist1->GetXaxis()->FindBin(l1pt)));
            int etabin1 = std::max(1, std::min(hist1->GetNbinsY(), hist1->GetYaxis()->FindBin(std::abs(l1eta))));
            double fr1 = hist1->GetBinContent(ptbin1,etabin1);
            TH2 *hist2 = (abs(l2pdgId) == 11 ? FRi_el[iFR] : FRi_mu[iFR]);
            int ptbin2  = std::max(1, std::min(hist2->GetNbinsX(), hist2->GetXaxis()->FindBin(l2pt)));
            int etabin2 = std::max(1, std::min(hist2->GetNbinsY(), hist2->GetYaxis()->FindBin(std::abs(l2eta))));
            double fr2 = hist2->GetBinContent(ptbin2,etabin2);
            return -fr1*fr2/((1-fr1)*(1-fr2));
        }
        default: return 0;
    }
}

float fakeRateWeight_2lssCB(float l1pt, float l1eta, int l1pdgId, float l1relIso,
                            float l2pt, float l2eta, int l2pdgId, float l2relIso, float WP) 
{
    return fakeRateWeight_2lssCB_i(l1pt, l1eta, l1pdgId, l1relIso,
                            l2pt, l2eta, l2pdgId, l2relIso, WP, 0);
}


float fakeRateWeight_2lssSyst(float l1pt, float l1eta, int l1pdgId, float l1mva,
                         float l2pt, float l2eta, int l2pdgId, float l2mva, float WP, 
                         float mu_barrel_lowpt, float mu_barrel_highpt, float mu_endcap_lowpt, float mu_endcap_highpt,
                         float el_cb_lowpt, float el_cb_highpt, float el_fb_lowpt, float el_fb_highpt, float el_endcap_lowpt, float el_endcap_highpt)
{
    /// 2 pass: weight  0
    /// 1 fail: weight +f/(1-f)
    /// 2 fail: weight -f*f/(1-f)(1-f)
    //  so, just multiply up factors of -f/(1-f) for each failure
    float mvas[]={l1mva, l2mva};
    float pts[]={l1pt, l2pt};
    float etas[]={fabs(l1eta), fabs(l2eta)};
    int pdgids[]={l1pdgId, l2pdgId};
    float ret = -1.0f;
    for (unsigned int i = 0; i < 2 ; ++i) {
        if (mvas[i] < WP) {
	    TH2 *hist = (abs(pdgids[i]) == 11 ? FR_el : FR_mu);
            int ptbin  = std::max(1, std::min(hist->GetNbinsX(), hist->GetXaxis()->FindBin(pts[i])));
            int etabin = std::max(1, std::min(hist->GetNbinsY(), hist->GetYaxis()->FindBin(etas[i])));
            double fr = hist->GetBinContent(ptbin,etabin);
            if (abs(pdgids[i]) == 11) fr *= ( std::abs(etas[i]) < 0.8 ? (pts[i] < 30 ? el_cb_lowpt : el_cb_highpt) :
                                             (std::abs(etas[i]) < 1.5 ? (pts[i] < 30 ? el_fb_lowpt : el_fb_highpt) :
                                                                        (pts[i] < 30 ? el_endcap_lowpt : el_endcap_highpt) ));
            else /*==13*/             fr *= (std::abs(etas[i]) < 1.5 ?  (pts[i] < 30 ? mu_barrel_lowpt : mu_barrel_highpt) :
                                                                        (pts[i] < 30 ? mu_endcap_lowpt : mu_endcap_highpt) );
            ret *= -fr/(1.0f-fr);
        }
    }
    if (ret == -1.0f) ret = 0.0f;
    return ret;

}
float fakeRateWeight_2lssBCat(float l1pt, float l1eta, int l1pdgId, float l1mva,
                         float l2pt, float l2eta, int l2pdgId, float l2mva, float WP, 
                         int nBJetMedium25, float scaleMuBL, float scaleMuBT, float scaleElBL, float scaleElBT)
{
    /// 2 pass: weight  0
    /// 1 fail: weight +f/(1-f)
    /// 2 fail: weight -f*f/(1-f)(1-f)
    //  so, just multiply up factors of -f/(1-f) for each failure
    float mvas[]={l1mva, l2mva};
    float pts[]={l1pt, l2pt};
    float etas[]={fabs(l1eta), fabs(l2eta)};
    int pdgids[]={l1pdgId, l2pdgId};
    float ret = -1.0f;
    for (unsigned int i = 0; i < 2 ; ++i) {
        if (mvas[i] < WP) {
	    TH2 *hist = (abs(pdgids[i]) == 11 ? (nBJetMedium25 > 1 ? FR2_el : FR_el):
                                                (nBJetMedium25 > 1 ? FR2_mu : FR_mu));
            int ptbin  = std::max(1, std::min(hist->GetNbinsX(), hist->GetXaxis()->FindBin(pts[i])));
            int etabin = std::max(1, std::min(hist->GetNbinsY(), hist->GetYaxis()->FindBin(etas[i])));
            double fr = hist->GetBinContent(ptbin,etabin);
            fr *= (nBJetMedium25 > 1 ? (abs(pdgids[i]) == 11 ? scaleElBT : scaleMuBT) : 
                                       (abs(pdgids[i]) == 11 ? scaleElBL : scaleMuBL) );
            ret *= -fr/(1.0f-fr);
        }
    }
    if (ret == -1.0f) ret = 0.0f;
    return ret;
}

float fakeRateWeight_2lssBCatSB(float l1pt, float l1eta, int l1pdgId, float l1mva,
                         float l2pt, float l2eta, int l2pdgId, float l2mva, float WP, float SBlow, float SBhigh,
                         int nBJetMedium25)
{
    float mvas[]={l1mva, l2mva};
    float pts[]={l1pt, l2pt};
    float etas[]={fabs(l1eta), fabs(l2eta)};
    int pdgids[]={l1pdgId, l2pdgId};
    float ret = -1.0f;
    for (unsigned int i = 0; i < 2 ; ++i) {
        if (mvas[i] > WP) {
            continue;
        } else if (SBlow < mvas[i] && mvas[i] < SBhigh) {
	    TH2 *hist = (abs(pdgids[i]) == 11 ? (nBJetMedium25 > 1 ? FR2_el : FR_el):
                                                (nBJetMedium25 > 1 ? FR2_mu : FR_mu));
            int ptbin  = std::max(1, std::min(hist->GetNbinsX(), hist->GetXaxis()->FindBin(pts[i])));
            int etabin = std::max(1, std::min(hist->GetNbinsY(), hist->GetYaxis()->FindBin(etas[i])));
            double fr = hist->GetBinContent(ptbin,etabin);
            ret *= -fr/max(1.0f-fr,0.5);
        } else {
            ret = 0.0f; break;
        }
    }
    if (ret == -1.0f) ret = 0.0f;
    return ret;
}
float fakeRateWeight_2lssBCatX(float l1pt, float l1eta, int l1pdgId, float l1mva,
                         float l2pt, float l2eta, int l2pdgId, float l2mva, float WPlow, float WPhigh,
                         int nBJetMedium25)
{
    float mvas[]={l1mva, l2mva};
    float pts[]={l1pt, l2pt};
    float etas[]={fabs(l1eta), fabs(l2eta)};
    int pdgids[]={l1pdgId, l2pdgId};
    float ret = -1.0f;
    int npass = 0;
    for (unsigned int i = 0; i < 2 ; ++i) {
        if (mvas[i] > WPhigh) {
            npass++; continue;
        } else if (mvas[i] > WPlow) {
            continue;
        } else  {
	    TH2 *hist = (abs(pdgids[i]) == 11 ? (nBJetMedium25 > 1 ? FR2_el : FR_el):
                                                (nBJetMedium25 > 1 ? FR2_mu : FR_mu));
            int ptbin  = std::max(1, std::min(hist->GetNbinsX(), hist->GetXaxis()->FindBin(pts[i])));
            int etabin = std::max(1, std::min(hist->GetNbinsY(), hist->GetYaxis()->FindBin(etas[i])));
            double fr = hist->GetBinContent(ptbin,etabin);
            ret *= -fr/std::max(1.0f-fr,0.5);
        }
    }
    if (ret == -1.0f) ret = 0.0f;
    return ret;
}




float fakeRateWeight_2lssMuIDCat(float l1pt, float l1eta, int l1pdgId, float l1mva, float l1tightId,
                         float l2pt, float l2eta, int l2pdgId, float l2mva, float l2tightId, float WP)
{
    /// 2 pass: weight  0
    /// 1 fail: weight +f/(1-f)
    /// 2 fail: weight -f*f/(1-f)(1-f)
    //  so, just multiply up factors of -f/(1-f) for each failure
    float mvas[]={l1mva, l2mva};
    float pts[]={l1pt, l2pt};
    float etas[]={fabs(l1eta), fabs(l2eta)};
    float tightIds[]={l1tightId, l2tightId};
    int pdgids[]={l1pdgId, l2pdgId};
    float ret = -1.0f;
    for (unsigned int i = 0; i < 2 ; ++i) {
        if (mvas[i] < WP) {
	    TH2 *hist = (abs(pdgids[i]) == 11 ? FR_el: (tightIds[i] > 0 ? FR_mu : FR2_mu));
            int ptbin  = std::max(1, std::min(hist->GetNbinsX(), hist->GetXaxis()->FindBin(pts[i])));
            int etabin = std::max(1, std::min(hist->GetNbinsY(), hist->GetYaxis()->FindBin(etas[i])));
            double fr = hist->GetBinContent(ptbin,etabin);
            ret *= -fr/(1.0f-fr);
        }
    }
    if (ret == -1.0f) ret = 0.0f;
    return ret;
}

float fakeRateWeight_2lssCB_ptRel2D(float l1pt, float l1eta, int l1pdgId, float l1relIso, float l1ptRel,
                                    float l2pt, float l2eta, int l2pdgId, float l2relIso, float l2ptRel, float WPIsoL, float WPIsoT, float WPPtRelL, float WPPtRelT) 
{
    float relIsos[]={l1relIso, l2relIso};
    float ptRels[]={l1ptRel, l2ptRel};
    float pts[]={l1pt, l2pt};
    float etas[]={fabs(l1eta), fabs(l2eta)};
    int pdgids[]={l1pdgId, l2pdgId};
    float ret = 0.f;
    int npass = 0;
    for (unsigned int i = 0; i < 2 ; ++i) {
        if (relIsos[i] < WPIsoT || ptRels[i] > WPPtRelT) { 
            npass++; continue; 
        }
        if (relIsos[i] < WPIsoL && ptRels[i] <= WPPtRelL) {
            // iso sideband
	    TH2 *hist = abs(pdgids[i]) == 11 ? FR2_el : FR2_mu;
            int ptbin  = std::max(1, std::min(hist->GetNbinsX(), hist->GetXaxis()->FindBin(pts[i])));
            int etabin = std::max(1, std::min(hist->GetNbinsY(), hist->GetYaxis()->FindBin(etas[i])));
            double fr = hist->GetBinContent(ptbin,etabin);
            ret += fr/(1.0f-fr);
        }
        if (ptRels[i] > WPPtRelL) {
            // ptrel sideband
	    TH2 *hist = abs(pdgids[i]) == 11 ? FR3_el : FR3_mu;
            int ptbin  = std::max(1, std::min(hist->GetNbinsX(), hist->GetXaxis()->FindBin(pts[i])));
            int etabin = std::max(1, std::min(hist->GetNbinsY(), hist->GetYaxis()->FindBin(etas[i])));
            double fr = hist->GetBinContent(ptbin,etabin);
            ret += fr/(1.0f-fr);
        }
    }
    if (npass != 1) ret = 0.0f;
    return ret;
}



float fakeRateWeight_3lMuIDCat(float l1pt, float l1eta, int l1pdgId, float l1mva, float l1tightId,
                        float l2pt, float l2eta, int l2pdgId, float l2mva, float l2tightId,
                        float l3pt, float l3eta, int l3pdgId, float l3mva, float l3tightId, float WP)
{
    float mvas[]={l1mva, l2mva, l3mva};
    float pts[]={l1pt, l2pt, l3pt};
    float etas[]={fabs(l1eta), fabs(l2eta), fabs(l3eta)};
    int pdgids[]={l1pdgId, l2pdgId, l3pdgId};
    float tightIds[]={l1tightId, l2tightId,l3tightId};
    float ret = -1.0f;
    for (unsigned int i = 0; i < 3 ; ++i) {
        if (mvas[i] < WP) {
	    TH2 *hist = (abs(pdgids[i]) == 11 ? FR_el: (tightIds[i] > 0 ? FR_mu : FR2_mu));
            int ptbin  = std::max(1, std::min(hist->GetNbinsX(), hist->GetXaxis()->FindBin(pts[i])));
            int etabin = std::max(1, std::min(hist->GetNbinsY(), hist->GetYaxis()->FindBin(etas[i])));
            double fr = hist->GetBinContent(ptbin,etabin);
            ret *= -fr/(1.0f-fr);
        }
    }
    if (ret == -1.0f) ret = 0.0f;
    return ret;
}

bool passND_LooseDen(float l1pt, float l1eta, int l1pdgId, float relIso, float dxy, float dz, float tightId) 
{
    if (fabs(l1pdgId) == 13) {
        return l1pt >= 10;
    } else {
        return l1pt >= 10 && (fabs(l1eta)<1.4442 || fabs(l1eta)>1.5660);
    }
}

bool passND_Loose(float l1pt, float l1eta, int l1pdgId, float relIso, float dxy, float dz, float tightId) 
{
    if (fabs(l1pdgId) == 13) {
        return l1pt >= 10 && 
               relIso < 0.2;
    } else {
        return l1pt >= 10 && (fabs(l1eta)<1.4442 || fabs(l1eta)>1.5660) &&
               tightId > 0.5 && relIso < 0.2 && fabs(dxy) < 0.04;
    }
}

bool passND_TightDen(float l1pt, float l1eta, int l1pdgId, float relIso, float dxy, float dz, float tightId) 
{
    if (fabs(l1pdgId) == 13) {
        return l1pt >= 20 && fabs(l1eta) <= 2.1;
    } else {
        return l1pt >= 20 && (fabs(l1eta)<1.4442 || fabs(l1eta)>1.5660);
    }
}

bool passND_Tight(float l1pt, float l1eta, int l1pdgId, float relIso, float dxy, float dz, float tightId) 
{
    if (fabs(l1pdgId) == 13) {
        return l1pt >= 20 && fabs(l1eta) <= 2.1 && 
               tightId != 0 && relIso < 0.12 && fabs(dxy) < 0.2 && fabs(dz) < 0.5;
    } else {
        return l1pt >= 20 && (fabs(l1eta)<1.4442 || fabs(l1eta)>1.5660) &&
               tightId > 0.5 && relIso < 0.1 && fabs(dxy) < 0.02;
    }
}

bool passEgammaTightMVA(float pt, float eta, float tightid) {
    if (fabs(eta) > 0.8) {
        return (pt > 20 ? (tightid > 0.94) : (tightid > 0.00));
    } else if (fabs(eta) < 1.479) {
        return (pt > 20 ? (tightid > 0.85) : (tightid > 0.10));
    } else {
        return (pt > 20 ? (tightid > 0.92) : (tightid > 0.062));
    }
}

float fakeRateWeight_2lss_ND(float l1pt, float l1eta, int l1pdgId, float l1relIso, float l1dxy, float l1dz, float l1tightId,
                          float l2pt, float l2eta, int l2pdgId, float l2relIso, float l2dxy, float l2dz, float l2tightId, int WP) 
{
    switch (WP) {
        case 11: {// loose-loose
            bool l1L = passND_Loose(l1pt, l1eta, l1pdgId, l1relIso, l1dxy, l1dz, l1tightId);
            bool l2L = passND_Loose(l2pt, l2eta, l2pdgId, l2relIso, l2dxy, l2dz, l2tightId);
            TH2 *hist1 = (abs(l1pdgId) == 11 ? FR_el : FR_mu);
            int ptbin1  = std::max(1, std::min(hist1->GetNbinsX(), hist1->GetXaxis()->FindBin(l1pt)));
            int etabin1 = std::max(1, std::min(hist1->GetNbinsY(), hist1->GetYaxis()->FindBin(std::abs(l1eta))));
            double fr1 = hist1->GetBinContent(ptbin1,etabin1);
            TH2 *hist2 = (abs(l2pdgId) == 11 ? FR_el : FR_mu);
            int ptbin2  = std::max(1, std::min(hist2->GetNbinsX(), hist2->GetXaxis()->FindBin(l2pt)));
            int etabin2 = std::max(1, std::min(hist2->GetNbinsY(), hist2->GetYaxis()->FindBin(std::abs(l2eta))));
            double fr2 = hist2->GetBinContent(ptbin2,etabin2);
            if      ( l1L &&  l2L) return 0;
            else if ( l1L && !l2L) return fr2/(1-fr2);
            else if (!l1L &&  l2L) return fr1/(1-fr1);
            else if (!l1L && !l2L) return -fr1*fr2/((1-fr1)*(1-fr2));
        }; 
        case 22: {// tight-tight 
            bool l1T = passND_Tight(l1pt, l1eta, l1pdgId, l1relIso, l1dxy, l1dz, l1tightId);
            bool l2T = passND_Tight(l2pt, l2eta, l2pdgId, l2relIso, l2dxy, l2dz, l2tightId);
            TH2 *hist1 = (abs(l1pdgId) == 11 ? FR2_el : FR2_mu);
            int ptbin1  = std::max(1, std::min(hist1->GetNbinsX(), hist1->GetXaxis()->FindBin(l1pt)));
            int etabin1 = std::max(1, std::min(hist1->GetNbinsY(), hist1->GetYaxis()->FindBin(std::abs(l1eta))));
            double fr1 = hist1->GetBinContent(ptbin1,etabin1);
            TH2 *hist2 = (abs(l2pdgId) == 11 ? FR2_el : FR2_mu);
            int ptbin2  = std::max(1, std::min(hist2->GetNbinsX(), hist2->GetXaxis()->FindBin(l2pt)));
            int etabin2 = std::max(1, std::min(hist2->GetNbinsY(), hist2->GetYaxis()->FindBin(std::abs(l2eta))));
            double fr2 = hist2->GetBinContent(ptbin2,etabin2);
            if      ( l1T &&  l2T) return 0;
            else if ( l1T && !l2T) return fr2/(1-fr2);
            else if (!l1T &&  l2T) return fr1/(1-fr1);
            else if (!l1T && !l2T) return -fr1*fr2/((1-fr1)*(1-fr2));
        }; 
        default: {
            static int _once = 0;
            if (_once++ == 0) { std::cerr << "ERROR, unknown WP " << WP << std::endl; }
        }

    }
    return 0;
}

float chargeFlipWeight_2lss(float l1pt, float l1eta, int l1pdgId, 
                             float l2pt, float l2eta, int l2pdgId) 
{
    if (l1pdgId * l2pdgId > 0) return 0.;
    double w = 0;
    if (abs(l1pdgId) == 11) {
        int ptbin  = std::max(1, std::min(QF_el->GetNbinsX(), QF_el->GetXaxis()->FindBin(l1pt)));
        int etabin = std::max(1, std::min(QF_el->GetNbinsY(), QF_el->GetYaxis()->FindBin(std::abs(l1eta))));
        w += QF_el->GetBinContent(ptbin,etabin);
    }
    if (abs(l2pdgId) == 11) {
        int ptbin  = std::max(1, std::min(QF_el->GetNbinsX(), QF_el->GetXaxis()->FindBin(l2pt)));
        int etabin = std::max(1, std::min(QF_el->GetNbinsY(), QF_el->GetYaxis()->FindBin(std::abs(l2eta))));
        w += QF_el->GetBinContent(ptbin,etabin);
    }
    return w;
}

float chargeFlipBin_2lss(float l1pt, float l1eta) {
    if (std::abs(l1eta) < 1.479) {
        return (l1pt < 20 ? 0 : (l1pt < 50 ? 1 : 2));
    } else {
        return (l1pt < 20 ? 3 : (l1pt < 50 ? 4 : 5));
    }
}


float fakeRateWeight_3lSyst(float l1pt, float l1eta, int l1pdgId, float l1mva,
                        float l2pt, float l2eta, int l2pdgId, float l2mva,
                        float l3pt, float l3eta, int l3pdgId, float l3mva,
                        float WP,
                        float mu_barrel_lowpt, float mu_barrel_highpt, float mu_endcap_lowpt, float mu_endcap_highpt,
                        float el_cb_lowpt, float el_cb_highpt, float el_fb_lowpt, float el_fb_highpt, float el_endcap_lowpt, float el_endcap_highpt)
{
    /// 3 pass: weight  0
    /// 1 fail: weight +f/(1-f)
    /// 2 fail: weight -f*f/(1-f)(1-f)
    //  3 fail: weight +f*f*f/((1-f)(1-f)(1-f)
    //  so, just multiply up factors of -f/(1-f) for each failure
    float mvas[]={l1mva, l2mva, l3mva};
    float pts[]={l1pt, l2pt, l3pt};
    float etas[]={fabs(l1eta), fabs(l2eta), fabs(l3eta)};
    int pdgids[]={l1pdgId, l2pdgId, l3pdgId};
    float ret = -1.0f;
    for (unsigned int i = 0; i < 3 ; ++i) {
        if (mvas[i] < WP) {
	    TH2 *hist = (abs(pdgids[i]) == 11 ? FR_el : FR_mu);
            int ptbin  = std::max(1, std::min(hist->GetNbinsX(), hist->GetXaxis()->FindBin(pts[i])));
            int etabin = std::max(1, std::min(hist->GetNbinsY(), hist->GetYaxis()->FindBin(etas[i])));
            double fr = hist->GetBinContent(ptbin,etabin);
            if (abs(pdgids[i]) == 11) fr *= ( std::abs(etas[i]) < 0.8 ? (pts[i] < 20 ? el_cb_lowpt : el_cb_highpt) :
                                             (std::abs(etas[i]) < 1.5 ? (pts[i] < 20 ? el_fb_lowpt : el_fb_highpt) :
                                                                        (pts[i] < 20 ? el_endcap_lowpt : el_endcap_highpt) ));
            else /*==13*/             fr *= (std::abs(etas[i]) < 1.5 ?  (pts[i] < 20 ? mu_barrel_lowpt : mu_barrel_highpt) :
                                                                        (pts[i] < 20 ? mu_endcap_lowpt : mu_endcap_highpt) );
            ret *= -fr/(1.0f-fr);
        }
    }
    if (ret == -1.0f) ret = 0.0f;
    return ret;
}

float fakeRateWeight_3l(float l1pt, float l1eta, int l1pdgId, float l1mva,
                        float l2pt, float l2eta, int l2pdgId, float l2mva,
                        float l3pt, float l3eta, int l3pdgId, float l3mva,
                        float WP)
{
    /// 3 pass: weight  0
    /// 1 fail: weight +f/(1-f)
    /// 2 fail: weight -f*f/(1-f)(1-f)
    //  3 fail: weight +f*f*f/((1-f)(1-f)(1-f)
    //  so, just multiply up factors of -f/(1-f) for each failure
    float mvas[]={l1mva, l2mva, l3mva};
    float pts[]={l1pt, l2pt, l3pt};
    float etas[]={fabs(l1eta), fabs(l2eta), fabs(l3eta)};
    int pdgids[]={l1pdgId, l2pdgId, l3pdgId};
    float ret = -1.0f;
    for (unsigned int i = 0; i < 3 ; ++i) {
        if (mvas[i] < WP) {
	    TH2 *hist = (abs(pdgids[i]) == 11 ? FR_el : FR_mu);
            int ptbin  = std::max(1, std::min(hist->GetNbinsX(), hist->GetXaxis()->FindBin(pts[i])));
            int etabin = std::max(1, std::min(hist->GetNbinsY(), hist->GetYaxis()->FindBin(etas[i])));
            double fr = hist->GetBinContent(ptbin,etabin);
            ret *= -fr/(1.0f-fr);
        }
    }
    if (ret == -1.0f) ret = 0.0f;
    return ret;
}

float fakeRateWeight_3lBCat(float l1pt, float l1eta, int l1pdgId, float l1mva,
                        float l2pt, float l2eta, int l2pdgId, float l2mva,
                        float l3pt, float l3eta, int l3pdgId, float l3mva,
                        float WP, int nBJetMedium25, float scaleMuBL, float scaleMuBT, float scaleElBL, float scaleElBT)
{
    /// 3 pass: weight  0
    /// 1 fail: weight +f/(1-f)
    /// 2 fail: weight -f*f/(1-f)(1-f)
    //  3 fail: weight +f*f*f/((1-f)(1-f)(1-f)
    //  so, just multiply up factors of -f/(1-f) for each failure
    float mvas[]={l1mva, l2mva, l3mva};
    float pts[]={l1pt, l2pt, l3pt};
    float etas[]={fabs(l1eta), fabs(l2eta), fabs(l3eta)};
    int pdgids[]={l1pdgId, l2pdgId, l3pdgId};
    float ret = -1.0f;
    for (unsigned int i = 0; i < 3 ; ++i) {
        if (mvas[i] < WP) {
	    TH2 *hist = (abs(pdgids[i]) == 11 ? (nBJetMedium25 > 1 ? FR2_el : FR_el):
                                                (nBJetMedium25 > 1 ? FR2_mu : FR_mu));
            int ptbin  = std::max(1, std::min(hist->GetNbinsX(), hist->GetXaxis()->FindBin(pts[i])));
            int etabin = std::max(1, std::min(hist->GetNbinsY(), hist->GetYaxis()->FindBin(etas[i])));
            double fr = hist->GetBinContent(ptbin,etabin);
            fr *= (nBJetMedium25 > 1 ? (abs(pdgids[i]) == 11 ? scaleElBT : scaleMuBT) : 
                                       (abs(pdgids[i]) == 11 ? scaleElBL : scaleMuBL) );
            ret *= -fr/(1.0f-fr);
        }
    }
    if (ret == -1.0f) ret = 0.0f;
    return ret;
}

float fakeRateWeight_3lCB(float l1pt, float l1eta, int l1pdgId, float l1relIso,
                        float l2pt, float l2eta, int l2pdgId, float l2relIso,
                        float l3pt, float l3eta, int l3pdgId, float l3relIso,
                        float WP)
{
    float relIsos[]={l1relIso, l2relIso, l3relIso};
    float pts[]={l1pt, l2pt, l3pt};
    float etas[]={fabs(l1eta), fabs(l2eta), fabs(l3eta)};
    int pdgids[]={l1pdgId, l2pdgId, l3pdgId};
    float ret = -1.0f;
    for (unsigned int i = 0; i < 3 ; ++i) {
        if (relIsos[i] > WP) {
	    TH2 *hist = (abs(pdgids[i]) == 11 ? FR_el : FR_mu);
            int ptbin  = std::max(1, std::min(hist->GetNbinsX(), hist->GetXaxis()->FindBin(pts[i])));
            int etabin = std::max(1, std::min(hist->GetNbinsY(), hist->GetYaxis()->FindBin(etas[i])));
            double fr = hist->GetBinContent(ptbin,etabin);
            ret *= -fr/(1.0f-fr);
        }
    }
    if (ret == -1.0f) ret = 0.0f;
    return ret;
}




float fakeRateWeight_4l_2wp(float l1pt, float l1eta, int l1pdgId, float l1mva,
                        float l2pt, float l2eta, int l2pdgId, float l2mva,
                        float l3pt, float l3eta, int l3pdgId, float l3mva,
                        float l4pt, float l4eta, int l4pdgId, float l4mva,
                        float WP, float WP2)
{
    /// 4 pass: weight  0
    /// 1 fail: weight +f/(1-f)
    /// 2 fail: weight -f*f/(1-f)(1-f)
    //  3 fail: weight +f*f*f/((1-f)(1-f)(1-f)
    //  so, just multiply up factors of -f/(1-f) for each failure
    //  hope it works also for 4l....
    float mvas[]={l1mva-WP, l2mva-WP, l3mva-WP2, l4mva-WP2};
    float pts[]={l1pt, l2pt, l3pt, l4pt};
    float etas[]={fabs(l1eta), fabs(l2eta), fabs(l3eta), fabs(l4eta)};
    int pdgids[]={l1pdgId, l2pdgId, l3pdgId, l4pdgId};
    float ret = -1.0f;
    int ifail = 0;
    for (unsigned int i = 0; i < 4 ; ++i) {
        if (mvas[i] < 0) {
            ifail++;
	    TH2 *hist = (i <= 1 ? (abs(pdgids[i]) == 11 ? FR_el : FR_mu) : (abs(pdgids[i]) == 11 ? FR2_el : FR2_mu));
            int ptbin  = std::max(1, std::min(hist->GetNbinsX(), hist->GetXaxis()->FindBin(pts[i])));
            int etabin = std::max(1, std::min(hist->GetNbinsY(), hist->GetYaxis()->FindBin(etas[i])));
            double fr = hist->GetBinContent(ptbin,etabin);
            ret *= -fr/(1.0f-fr);
        }
    }
    if (ifail > 2) return 0;
    if (ret == -1.0f) ret = 0.0f;
    return ret;
}

float fakeRateWeight_4l_2wp_nf(int nf, float l1pt, float l1eta, int l1pdgId, float l1mva,
                        float l2pt, float l2eta, int l2pdgId, float l2mva,
                        float l3pt, float l3eta, int l3pdgId, float l3mva,
                        float l4pt, float l4eta, int l4pdgId, float l4mva,
                        float WP, float WP2)
{
    /// 4 pass: weight  0
    /// 1 fail: weight +f/(1-f)
    /// 2 fail: weight -f*f/(1-f)(1-f)
    //  3 fail: weight +f*f*f/((1-f)(1-f)(1-f)
    //  so, just multiply up factors of -f/(1-f) for each failure
    //  hope it works also for 4l....
    float mvas[]={l1mva-WP, l2mva-WP, l3mva-WP2, l4mva-WP2};
    float pts[]={l1pt, l2pt, l3pt, l4pt};
    float etas[]={fabs(l1eta), fabs(l2eta), fabs(l3eta), fabs(l4eta)};
    int pdgids[]={l1pdgId, l2pdgId, l3pdgId, l4pdgId};
    float ret = (nf == 1 ? 0.5f : 1.0f);
    int ifail = 0;
    for (unsigned int i = 0; i < 4 ; ++i) {
        if (mvas[i] < 0) {
            ifail++;
	    TH2 *hist = (i <= 1 ? (abs(pdgids[i]) == 11 ? FR_el : FR_mu) : (abs(pdgids[i]) == 11 ? FR2_el : FR2_mu));
            int ptbin  = std::max(1, std::min(hist->GetNbinsX(), hist->GetXaxis()->FindBin(pts[i])));
            int etabin = std::max(1, std::min(hist->GetNbinsY(), hist->GetYaxis()->FindBin(etas[i])));
            double fr = hist->GetBinContent(ptbin,etabin);
            ret *= fr/(1.0f-fr);
        }
    }
    if (ifail != nf) return 0;
    return ret;
}

float fakeRate_flavour_2l_19_lead(float l1pt, float l1eta, int l1pdgId, 
                                 float l2pt, float l2eta, int l2pdgId, int histo) 
{
    if (l1pdgId == -l2pdgId) return 0.;
    switch (histo) {
        case 0: return 1.;
        case 1:
        case 2:
             TH2 *num = (histo <= 1 ? (abs(l1pdgId) == 11 ? FR_el : FR_mu) : (abs(l1pdgId) == 11 ? FR2_el : FR2_mu));
             TH2 *den = (histo <= 1 ? (abs(l1pdgId) == 11 ? FR_mu : FR_el) : (abs(l1pdgId) == 11 ? FR2_mu : FR2_el));
             int ptnum  = std::max(1, std::min(num->GetNbinsX(), num->GetXaxis()->FindBin(l1pt)));
             int etanum = std::max(1, std::min(num->GetNbinsY(), num->GetYaxis()->FindBin(fabs(l1eta))));
             int ptden  = std::max(1, std::min(den->GetNbinsX(), den->GetXaxis()->FindBin(l1pt)));
             int etaden = std::max(1, std::min(den->GetNbinsY(), den->GetYaxis()->FindBin(fabs(l1eta))));
             if (den->GetBinContent(ptden,etaden) == 0) return 1.0;
             return num->GetBinContent(ptnum,etanum)/den->GetBinContent(ptden,etaden);
    }
    return 0.;
}


float fakeRateBin_Muons(float pt, float eta) { // 0 .. 49
    if (pt >= 50) pt = 49.9;
    eta = fabs(eta); if (eta >= 2.5) eta = 2.499;
    int ieta = floor(eta/0.5); // 0 .. 4;
    int ipt  = floor(pt/5.0); // 0 .. 9;
    if (ipt == 8) ipt = 7; // now merge the 40-45 into the 35-40;
    return ipt*5 + ieta + 0.5; // make sure we end in the bin center
}
float fakeRateBin_Muons_eta(float bin) {
    int ibin = floor(bin);
    return (ibin % 5)*0.5 + 0.25;
}
float fakeRateBin_Muons_pt(float bin) {
    int ibin = floor(bin);
    return (ibin/5)*5.0 + 2.5;
}

namespace WP {
    enum WPId { V=0, VL=0, VVL=-1, L=1, M=2, T=3, VT=4, HT=5 } ;
}
float multiIso_singleWP(float LepGood_miniRelIso, float LepGood_jetPtRatio, float LepGood_jetPtRel, WP::WPId wp) {
    switch (wp) {
        case WP::HT: return LepGood_miniRelIso < 0.05  && (LepGood_jetPtRatio>0.725 || LepGood_jetPtRel>8   );
        case WP::VT: return LepGood_miniRelIso < 0.075 && (LepGood_jetPtRatio>0.725 || LepGood_jetPtRel>7   );
        case WP::T:  return LepGood_miniRelIso < 0.10  && (LepGood_jetPtRatio>0.700 || LepGood_jetPtRel>7   );
        case WP::M:  return LepGood_miniRelIso < 0.14  && (LepGood_jetPtRatio>0.68  || LepGood_jetPtRel>6.7 );
        case WP::L:  return LepGood_miniRelIso < 0.22  && (LepGood_jetPtRatio>0.63  || LepGood_jetPtRel>6.0 );
        case WP::VVL: return LepGood_miniRelIso < 0.4;
        default:
            std::cerr << "Working point " << wp << " not implemented for multiIso_singleWP" << std::endl;
            abort();
    }
}
float multiIso_multiWP(int LepGood_pdgId, float LepGood_pt, float LepGood_eta, float LepGood_miniRelIso, float LepGood_jetPtRatio, float LepGood_jetPtRel, WP::WPId wp) {
    switch (wp) {
        case WP::VT: 
           return abs(LepGood_pdgId)==13 ? 
                    multiIso_singleWP(LepGood_miniRelIso,LepGood_jetPtRatio,LepGood_jetPtRel, WP::VT) :
                    multiIso_singleWP(LepGood_miniRelIso,LepGood_jetPtRatio,LepGood_jetPtRel, WP::HT) ;
        case WP::T:
           return abs(LepGood_pdgId)==13 ? 
                    multiIso_singleWP(LepGood_miniRelIso,LepGood_jetPtRatio,LepGood_jetPtRel, WP::T) :
                    multiIso_singleWP(LepGood_miniRelIso,LepGood_jetPtRatio,LepGood_jetPtRel, WP::VT) ;
        case WP::M:
           return abs(LepGood_pdgId)==13 ? 
                    multiIso_singleWP(LepGood_miniRelIso,LepGood_jetPtRatio,LepGood_jetPtRel, WP::M) :
                    multiIso_singleWP(LepGood_miniRelIso,LepGood_jetPtRatio,LepGood_jetPtRel, WP::T) ;
        case WP::VVL: return LepGood_miniRelIso < 0.4;
        default:
            std::cerr << "Working point " << wp << " not implemented for multiIso_multiWP" << std::endl;
            abort();
    }
}
float multiIso_multiWP(int LepGood_pdgId, float LepGood_pt, float LepGood_eta, float LepGood_miniRelIso, float LepGood_jetPtRatio, float LepGood_jetPtRel, int wp) {
    return multiIso_multiWP(LepGood_pdgId,LepGood_pt,LepGood_eta,LepGood_miniRelIso,LepGood_jetPtRatio,LepGood_jetPtRel,WP::WPId(wp));
}


void fakeRate() {}
