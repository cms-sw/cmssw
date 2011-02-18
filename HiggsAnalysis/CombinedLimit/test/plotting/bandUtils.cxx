#include <map>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <TTree.h>
#include <TFile.h>
#include <TGraphAsymmErrors.h>
#include <Math/ProbFunc.h>


// Maritz-Jarrett, JASA vol. 73 (1978)
// http://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/quantse.htm
double quantErr(size_t n, double *vals, double q) {
    int m = floor(q*n+0.5);
    double a = m-1;
    double b = n-m;
    double ninv = 1.0/n;
    double c1 = 0, c2 = 0;
    double last_cdf = 0;
    for (size_t i = 0; i < n; ++i) {
        double this_cdf = ROOT::Math::beta_cdf((i+1) * ninv, a, b);
        double w = this_cdf - last_cdf;
        c1 += w * vals[i];
        c2 += w * vals[i] * vals[i];
        last_cdf = this_cdf;
    }
    return sqrt(c2 - c1*c1);
}

enum BandType { Mean, Median, Quantile, Observed, Asimov, CountToys, MeanCPUTime, AdHoc };
double band_safety_crop = 0;
TGraphAsymmErrors *theBand(TFile *file, int doSyst, int whichChannel, BandType type, double width=0.68) {
    if (file == 0) return 0;
    TTree *t = (TTree *) file->Get("test");
    if (t == 0) { std::cerr << "TFile " << file->GetName() << " does not contain the tree" << std::endl; return 0; }
    Double_t mass, limit; Float_t t_cpu, t_real; Int_t syst, iChannel, iToy, iMass;
    t->SetBranchAddress("mh", &mass);
    t->SetBranchAddress("limit", &limit);
    if (t->GetBranch("t_cpu") != 0) {
        t->SetBranchAddress("t_cpu", &t_cpu);
        t->SetBranchAddress("t_real", &t_real);
    }
    t->SetBranchAddress("syst", &syst);
    t->SetBranchAddress("iChannel", &iChannel);
    t->SetBranchAddress("iToy", &iToy);

    std::map<int,std::vector<double> >  dataset;
    for (size_t i = 0, n = t->GetEntries(); i < n; ++i) {
        t->GetEntry(i);
        if (syst != doSyst)           continue;
        if (iChannel != whichChannel) continue;
        if      (type == Asimov)   { if (iToy != -1) continue; }
        else if (type == Observed) { if (iToy !=  0) continue; }
        else if (iToy <= 0) continue;
        if (limit == 0) continue; 
        iMass = int(mass);
        if (type == MeanCPUTime) { 
            if (limit < 0) continue; 
            limit = t_cpu; 
        }
        dataset[iMass].push_back(limit);
    }
    TGraphAsymmErrors *tge = new TGraphAsymmErrors(dataset.size());
    int ip = 0;
    for (std::map<int,std::vector<double> >::iterator it = dataset.begin(), ed = dataset.end(); it != ed; ++it, ++ip) {
        std::vector<double> &data = it->second;
        int nd = data.size();
        std::sort(data.begin(), data.end());
        double median = (data.size() % 2 == 0 ? 0.5*(data[nd/2]+data[nd/2+1]) : data[nd/2]);
        if (band_safety_crop > 0) {
            std::vector<double> data2;
            for (int j = 0; j < nd; ++j) {
                if (data[j] > median*band_safety_crop && data[j] < median/band_safety_crop) {
                    data2.push_back(data[j]);
                }
            }
            data2.swap(data);
            nd = data.size();
            median = (data.size() % 2 == 0 ? 0.5*(data[nd/2]+data[nd/2+1]) : data[nd/2]);
        }
        double mean = 0; for (int j = 0; j < nd; ++j) mean += data[j]; mean /= nd;
        double summer68 = data[floor(nd * 0.5*(1-width)+0.5)], winter68 =  data[std::min(int(floor(nd * 0.5*(1+width)+0.5)), nd-1)];
        double x = mean;
        switch (type) {
            case MeanCPUTime:
            case Mean: x = mean; break;
            case Median: x = median; break;
            case CountToys: x = summer68 = winter68 = nd; break;
            case Asimov: // mean (in case we did it more than once), with no band
                x = summer68 = winter68 = mean;
                break;
            case Observed:
                x = mean;
                if (nd <= 1) {
                    summer68 = winter68 = mean;
                } else { // if we have multiple, average and report rms (useful e.g. for MCMC)
                    double rms = 0;
                    for (int j = 0; j < nd; ++j) { rms += (mean-data[j])*(mean-data[j]); }
                    rms = sqrt(rms/(nd*(nd-1)));
                    summer68 = mean + rms;
                    winter68 = mean - rms;
                }
            case AdHoc:
                x = summer68 = winter68 = mean;
                break;
            case Quantile: // get the quantile equal to width, and it's uncertainty
                x = data[floor(nd*width+0.5)];
                summer68 = x - quantErr(nd, &data[0], width);
                winter68 = x + (x-summer68);
                break;

        }
        tge->SetPoint(ip, it->first, x);
        tge->SetPointError(ip, 0, 0, x-summer68, winter68-x);
    }
    return tge;
}

void theBand() {}
void makeBand(TDirectory *bands, TString name, TFile *file, int doSyst, int whichChannel, BandType type) {
    TString suffix = "";
    switch (type) {
        case Asimov:    suffix = "_asimov"; break;
        case Observed:  suffix = "_obs"; break;
        case Mean:      suffix = "_mean"; break;
        case Median:    suffix = "_median"; break;
        case CountToys: suffix = "_ntoys"; break;
        case MeanCPUTime: suffix = "_cputime"; break;
        case AdHoc:       suffix = ""; break;
        case Quantile:    suffix = ""; break;
    }
    if (!doSyst && (type != AdHoc)) suffix = "_nosyst"+suffix;
    if (type == Median || type == Mean) {
        TGraph *band68 = theBand(file, doSyst, whichChannel, type, 0.68);
        TGraph *band95 = theBand(file, doSyst, whichChannel, type, 0.95);
        if (band68 != 0 && band68->GetN() > 0) {
            band68->SetName(name+suffix);
            bands->WriteTObject(band68, name+suffix);
            band95->SetName(name+suffix+"_95");
            bands->WriteTObject(band95, name+suffix+"_95");
        } else {
            std::cout << "Band " << name+suffix << " missing" << std::endl;
        }
    } else {
        TGraph *band = theBand(file, doSyst, whichChannel, type);
        if (band != 0 && band->GetN() > 0) {
            band->SetName(name+suffix);
            bands->WriteTObject(band, name+suffix);
        } else {
            std::cout << "Band " << name+suffix << " missing" << std::endl;
        }
    }
}
void makeLine(TDirectory *bands, TString name, TString filename,  int doSyst, int whichChannel) {
    TFile *in = TFile::Open(filename);
    if (in == 0) { std::cerr << "Filename '" << filename << "' missing" << std::endl; return; }
    makeBand(bands, name, in, doSyst, whichChannel, AdHoc);
    in->Close();
}
void makeBands(TDirectory *bands, TString name, TString filename, int channel=0, bool quantiles=false) {
    TFile *in = TFile::Open(filename);
    if (in == 0) { std::cerr << "Filename " << filename << " missing" << std::endl; return; }
    for (int s = 0; s <= 1; ++s) {
        makeBand(bands, name, in, s, channel, Mean);
        makeBand(bands, name, in, s, channel, Median);
        makeBand(bands, name, in, s, channel, CountToys);
        makeBand(bands, name, in, s, channel, Observed);
    }
    if (quantiles) {
        double quants[5] = { 0.025, 0.16, 0.5, 0.84, 0.975 };
        for (int i = 0; i < 5; ++i) {
            for (int s = 0; s <= 1; ++s) {
                TGraph *band = theBand(in, s, channel, Quantile, quants[i]);
                TString qname = TString::Format("%s%s_quant%03d", name.Data(), (s ? "" : "_nosyst"), int(1000*quants[i]));
                if (band != 0 && band->GetN() != 0) {
                    band->SetName(qname);
                    bands->WriteTObject(band, qname);
                } else {
                    std::cout << "Band " << qname << " missing" << std::endl;
                }
            }
        }
    }
    in->Close();
}


void cutBand(TDirectory *bands, TString inName, TString outName, int mMin, int mMax) {
    TGraphAsymmErrors *b1 = (TGraphAsymmErrors *) bands->Get(inName);
    if (b1 == 0 || b1->GetN() == 0) return;
    TGraphAsymmErrors *b2 = new TGraphAsymmErrors();
    int n = b1->GetN(), m = 0;
    for (int i = 0; i < n; ++i) {
        if (mMin <= int(b1->GetX()[i]) && int(b1->GetX()[i]) <= mMax) {
            b2->Set(m+1);
            b2->SetPoint(m, b1->GetX()[i], b1->GetY()[i]);
            b2->SetPointError(m, b1->GetErrorXlow(i), b1->GetErrorXhigh(i),
                                 b1->GetErrorYlow(i), b1->GetErrorYhigh(i));
            m++;
        }
    }
    b2->SetName(outName);
    bands->WriteTObject(b2, outName);
}

void cutBands(TDirectory *bands, TString inName, TString outName, int mMin, int mMax) {
    cutBand(bands, inName+"_mean",   outName+"_mean",    mMin, mMax);
    cutBand(bands, inName+"_median", outName+"_median",  mMin, mMax);
    cutBand(bands, inName+"_mean_95",   outName+"_mean_95",    mMin, mMax);
    cutBand(bands, inName+"_median_95", outName+"_median_95",  mMin, mMax);
    cutBand(bands, inName+"_asimov",    outName+"_asimov",     mMin, mMax);
    cutBand(bands, inName+"_ntoys",     outName+"_ntoys",      mMin, mMax);

    cutBand(bands, inName+"_nosyst_mean",   outName+"_nosyst_mean",    mMin, mMax);
    cutBand(bands, inName+"_nosyst_median", outName+"_nosyst_median",  mMin, mMax);
    cutBand(bands, inName+"_nosyst_mean_95",   outName+"_nosyst_mean_95",    mMin, mMax);
    cutBand(bands, inName+"_nosyst_asimov",    outName+"_nosyst_asimov",     mMin, mMax);
    cutBand(bands, inName+"_nosyst_ntoys",     outName+"_nosyst_ntoys",      mMin, mMax);
}

void combineBand(TDirectory *in, TString band1, TString band2, TString comb) {
    TGraphAsymmErrors *b1 = (TGraphAsymmErrors *) in->Get(band1);
    TGraphAsymmErrors *b2 = (TGraphAsymmErrors *) in->Get(band2);
    if (b1 == 0 || b1->GetN() == 0) return;
    if (b2 == 0 || b2->GetN() == 0) return;
    int n = b1->GetN(), m = b2->GetN();
    int first = n, last = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (int(b1->GetX()[i]) == int(b2->GetX()[j])) {
                if (i < first) first = i;
                last = i;
            }
        }
    }
    TGraphAsymmErrors *bc = new TGraphAsymmErrors((first-1) + m + (n-last-1));
    bc->SetName(comb); 
    int k = 0;
    for (int i = 0; i < first; ++i, ++k) {
        bc->SetPoint(k, b1->GetX()[i], b1->GetY()[i]);
        bc->SetPointError(k, b1->GetErrorXlow(i), b1->GetErrorXhigh(i), 
                             b1->GetErrorYlow(i), b1->GetErrorYhigh(i));
    }
    for (int i = 0; i < m; ++i, ++k) {
        bc->SetPoint(k, b2->GetX()[i], b2->GetY()[i]);
        bc->SetPointError(k, b2->GetErrorXlow(i), b2->GetErrorXhigh(i), 
                             b2->GetErrorYlow(i), b2->GetErrorYhigh(i));
    }
    for (int i = last+1; i < n; ++i, ++k) {
        bc->SetPoint(k, b1->GetX()[i], b1->GetY()[i]);
        bc->SetPointError(k, b1->GetErrorXlow(i), b1->GetErrorXhigh(i), 
                             b1->GetErrorYlow(i), b1->GetErrorYhigh(i));
    }
    bc->SetName(comb);
    in->WriteTObject(bc, comb);
}
void combineBands(TDirectory *in, TString band1, TString band2, TString comb) {
    combineBand(in, band1+"_mean",   band2+"_mean",   comb+"_mean");
    combineBand(in, band1+"_median", band2+"_median", comb+"_median");
    combineBand(in, band1+"_mean_95",   band2+"_mean_95",   comb+"_mean_95");
    combineBand(in, band1+"_median_95", band2+"_median_95", comb+"_median_95");
    combineBand(in, band1+"_asimov",    band2+"_asimov",    comb+"_asimov");
    combineBand(in, band1+"_ntoys",    band2+"_ntoys",    comb+"_ntoys");

    combineBand(in, band1+"_nosyst_mean",   band2+"_nosyst_mean",   comb+"_nosyst_mean");
    combineBand(in, band1+"_nosyst_median", band2+"_nosyst_median", comb+"_nosyst_median");
    combineBand(in, band1+"_nosyst_mean_95",   band2+"_nosyst_mean_95",   comb+"_nosyst_mean_95");
    combineBand(in, band1+"_nosyst_median_95", band2+"_nosyst_median_95", comb+"_nosyst_median_95");
    combineBand(in, band1+"_nosyst_asimov",    band2+"_nosyst_asimov",    comb+"_nosyst_asimov");
    combineBand(in, band1+"_nosyst_ntoys",    band2+"_nosyst_ntoys",    comb+"_nosyst_ntoys");
}

void pasteBand(TDirectory *in, TString band1, TString band2, TString comb) {
    TGraphAsymmErrors *b1 = (TGraphAsymmErrors *) in->Get(band1);
    TGraphAsymmErrors *b2 = (TGraphAsymmErrors *) in->Get(band2);
    if (b1 == 0 || b1->GetN() == 0) return;
    if (b2 == 0 || b2->GetN() == 0) return;
    TGraphAsymmErrors *bc = new TGraphAsymmErrors(b1->GetN()+b2->GetN());
    bc->SetName(comb); 
    int k = 0, n = b1->GetN(), m = b2->GetN();
    for (int i = 0; i < n; ++i, ++k) {
        bc->SetPoint(k, b1->GetX()[i], b1->GetY()[i]);
        bc->SetPointError(k, b1->GetErrorXlow(i), b1->GetErrorXhigh(i), 
                             b1->GetErrorYlow(i), b1->GetErrorYhigh(i));
    }
    for (int i = 0; i < m; ++i, ++k) {
        bc->SetPoint(k, b2->GetX()[i], b2->GetY()[i]);
        bc->SetPointError(k, b2->GetErrorXlow(i), b2->GetErrorXhigh(i), 
                             b2->GetErrorYlow(i), b2->GetErrorYhigh(i));
    }
    bc->Sort();
    in->WriteTObject(bc, comb);
}
void pasteBands(TDirectory *in, TString band1, TString band2, TString comb) {
    pasteBand(in, band1+"_mean",   band2+"_mean",   comb+"_mean");
    pasteBand(in, band1+"_median", band2+"_median", comb+"_median");
    pasteBand(in, band1+"_mean_95",   band2+"_mean_95",   comb+"_mean_95");
    pasteBand(in, band1+"_median_95", band2+"_median_95", comb+"_median_95");
    pasteBand(in, band1+"_asimov",    band2+"_asimov",    comb+"_asimov");
    pasteBand(in, band1+"_ntoys",    band2+"_ntoys",    comb+"_ntoys");

    pasteBand(in, band1+"_nosyst_mean",   band2+"_nosyst_mean",   comb+"_nosyst_mean");
    pasteBand(in, band1+"_nosyst_median", band2+"_nosyst_median", comb+"_nosyst_median");
    pasteBand(in, band1+"_nosyst_mean_95",   band2+"_nosyst_mean_95",   comb+"_nosyst_mean_95");
    pasteBand(in, band1+"_nosyst_median_95", band2+"_nosyst_median_95", comb+"_nosyst_median_95");
    pasteBand(in, band1+"_nosyst_asimov",    band2+"_nosyst_asimov",    comb+"_nosyst_asimov");
    pasteBand(in, band1+"_nosyst_ntoys",    band2+"_nosyst_ntoys",    comb+"_nosyst_ntoys");
}

void stripPoint(TGraph *band, int m) {
    for (int i = 0, n = band->GetN(); i < n; ++i) {
        if (int(band->GetX()[i]) == m) {
            band->RemovePoint(i);
            return;
        }
    }
    if ((band->GetN() > 0) &&
        (band->GetX()[0] <= m) &&
        (band->GetX()[band->GetN()-1] >= m)) {
    }
}
void stripBand(TDirectory *in, TString band1, int m1, int m2=0, int m3=0, int m4=0, int m5=0) {
    TGraphAsymmErrors *band = (TGraphAsymmErrors *) in->Get(band1);
    if (band == 0 || band->GetN() == 0) return;
    if (m1) stripPoint(band,m1);
    if (m2) stripPoint(band,m2);
    if (m3) stripPoint(band,m3);
    if (m4) stripPoint(band,m4);
    if (m5) stripPoint(band,m5);
    in->WriteTObject(band, band->GetName(), "Overwrite");
}

void stripBands(TDirectory *in, TString band,  int m1, int m2=0, int m3=0, int m4=0, int m5=0) {
    stripBand(in, band+"_mean",      m1,m2,m3,m4,m5);
    stripBand(in, band+"_median",    m1,m2,m3,m4,m5);
    stripBand(in, band+"_mean_95",   m1,m2,m3,m4,m5);
    stripBand(in, band+"_median_95", m1,m2,m3,m4,m5);
    stripBand(in, band+"_asimov",    m1,m2,m3,m4,m5);
    stripBand(in, band+"_nosyst_mean",      m1,m2,m3,m4,m5);
    stripBand(in, band+"_nosyst_median",    m1,m2,m3,m4,m5);
    stripBand(in, band+"_nosyst_mean_95",   m1,m2,m3,m4,m5);
    stripBand(in, band+"_nosyst_median_95", m1,m2,m3,m4,m5);
    stripBand(in, band+"_nosyst_asimov",    m1,m2,m3,m4,m5);
}

void printBand(TDirectory *bands, TString who, FILE *fout, bool mean=true) {
    TGraph *mean68 = (TGraph*) bands->Get(who+(mean?"_mean":"_median"));
    TGraph *mean95 = (TGraph*) bands->Get(who+(mean?"_mean":"_median")+"_95");
    if (mean68 == 0) { std::cerr << "MISSING " << who << "_mean" << std::endl; return; }
    fprintf(fout, "%4s \t %7s  %7s  %7s  %7s  %7s\n", "mass", "-95%", "-68%", (mean ? "mean" : "median"), "+68%", "+95%");
    fprintf(fout,  "%5s\t %7s  %7s  %7s  %7s  %7s\n", "-----", "-----", "-----", "-----", "-----", "-----");
    for (int i = 0, n = mean68->GetN(); i < n; ++i) {
        fprintf(fout, "%4d \t %7.3f  %7.3f  %7.3f  %7.3f  %7.3f\n", 
            int(mean68->GetX()[i]),  
            mean68->GetY()[i]-mean95->GetErrorYlow(i), 
            mean68->GetY()[i]-mean68->GetErrorYlow(i), 
            mean68->GetY()[i],
            mean68->GetY()[i]+mean68->GetErrorYhigh(i),
            mean68->GetY()[i]+mean95->GetErrorYhigh(i)
        );
    }
}
void printQuantiles(TDirectory *bands, TString who, FILE *fout) {
    double quants[5] = { 0.025, 0.16, 0.5, 0.84, 0.975 };
    TGraphAsymmErrors *graphs[5];
    for (int i = 0; i < 5; ++i) {
        graphs[i] = (TGraphAsymmErrors *) bands->Get(who+TString::Format("_quant%03d", int(1000*quants[i])));
        if (graphs[i] == 0) { std::cout << "Missing quantile band for p = " << quants[i] << std::endl; return; }
    }
    fprintf(fout, "%4s \t %6s %5s   %6s %5s   %6s %5s   %6s %5s   %6s %5s\n", "mass", "-95%","err", "-68%","err", "median","err", "+68%","err", "+95%","err");
    fprintf(fout, "%4s \t %6s %5s   %6s %5s   %6s %5s   %6s %5s   %6s %5s\n", "-----", "-----", "-----", "-----", "-----", "-----","-----", "-----", "-----", "-----", "-----");
    for (int i = 0, n = graphs[0]->GetN(); i < n; ++i) {
        fprintf(fout, "%4d \t ", int(graphs[0]->GetX()[i]));
        for (int j = 0; j < 5; ++j) {
            fprintf(fout, "%6.2f %5.2f   ", graphs[j]->GetY()[i], graphs[j]->GetErrorYlow(i));
        }
        fprintf(fout, "\n");
    }
}
void printQuantiles(TDirectory *bands, TString who, TString fileName) {
    TGraph *mean68 = (TGraph*) bands->Get(who+"_quant025");
    if (mean68 == 0) { std::cerr << "MISSING " << who << "_quant025" << std::endl; return; }
    FILE *fout = fopen(fileName.Data(), "w");
    printQuantiles(bands,who,fout);
    fclose(fout);

}
void printBand(TDirectory *bands, TString who, TString fileName, bool mean=true) {
    TGraph *mean68 = (TGraph*) bands->Get(who+(mean?"_mean":"_median"));
    if (mean68 == 0) { std::cerr << "MISSING " << who << "_mean" << std::endl; return; }
    FILE *fout = fopen(fileName.Data(), "w");
    printBand(bands,who,fout,mean);
    fclose(fout);
}

void printLine(TDirectory *bands, TString who, TString fileName) {
    TGraph *mean = (TGraph*) bands->Get(who);
    if (mean == 0) { std::cerr << "MISSING " << who << std::endl; return; }
    FILE *fout = fopen(fileName.Data(), "w");
    fprintf(fout, "%4s \t %7s\n", "mass",  "value");
    fprintf(fout,  "%5s\t %7s\n", "-----", "-----");
    for (int i = 0, n = mean->GetN(); i < n; ++i) {
        fprintf(fout, "%4d \t %7.3f\n",  int(mean->GetX()[i]), mean->GetY()[i]);  
    }
    fclose(fout);
}


void importLine(TDirectory *bands, TString name, const char *fileName) {
    FILE *in = fopen(fileName, "r");
    if (in == 0) { std::cerr << "Cannot open " << fileName << std::endl; return; }
    TGraphAsymmErrors *inObs = new TGraphAsymmErrors(); inObs->SetName(name);
    float mH, yObs; 
    for (int n = 0; fscanf(in,"%f %f", &mH, &yObs) == 2; ++n) {
        inObs->SetPoint(n, mH, yObs);
    }
    bands->WriteTObject(inObs);
    fclose(in);
}

void importBands(TDirectory *bands, TString name, const char *fileName, bool hasObs = false, bool has95 = true) {
    FILE *in = fopen(fileName, "r");
    if (in == 0) { std::cerr << "Cannot open " << fileName << std::endl; return; }
    TGraphAsymmErrors *inObs = new TGraphAsymmErrors(); inObs->SetName(name+"_obs");
    TGraphAsymmErrors *in68  = new TGraphAsymmErrors();  in68->SetName(name);
    TGraphAsymmErrors *in95  = new TGraphAsymmErrors();  in95->SetName(name+"_95");
    float mH, yObs, yLL, yLo, y, yHi, yHH; 
    if (hasObs) {
        for (int n = 0; fscanf(in,"%f %f %f %f %f %f %f", &mH, &yObs, &yLL, &yLo, &y, &yHi, &yHH) == 7; ++n) {
            inObs->SetPoint(n, mH, yObs);
            in68->SetPoint(n, mH, y); in68->SetPointError(n, 0, 0, y-yLo, yHi-y);
            in95->SetPoint(n, mH, y); in95->SetPointError(n, 0, 0, y-yLL, yHH-y);
        }
    } else {
        if (has95) {
            for (int n = 0; fscanf(in,"%f %f %f %f %f %f", &mH, &yLL, &yLo, &y, &yHi, &yHH) == 6; ++n) {
                in68->SetPoint(n, mH, y); in68->SetPointError(n, 0, 0, y-yLo, yHi-y);
                in95->SetPoint(n, mH, y); in95->SetPointError(n, 0, 0, y-yLL, yHH-y);
            }
        } else {
            for (int n = 0; fscanf(in,"%f %f %f %f", &mH, &yLo, &y, &yHi) == 4; ++n) {
                in68->SetPoint(n, mH, y); in68->SetPointError(n, 0, 0, y-yLo, yHi-y);
            }
        }
    }
    bands->WriteTObject(in68);
    if (has95) bands->WriteTObject(in95);
    if (hasObs) bands->WriteTObject(inObs);
    fclose(in);
}

void bandUtils() {}
