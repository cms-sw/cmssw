#include <map>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <TTree.h>
#include <TFile.h>
#include <TF1.h>
#include <TMatrixDSym.h>
#include <TDecompBK.h>
#include <TVectorD.h>
#include <TGraphAsymmErrors.h>
#include <TGraphErrors.h>
#include <Math/ProbFunc.h>
#include <Math/QuantFuncMathCore.h>

double band_safety_crop = 0; 
bool use_precomputed_quantiles = false, precomputed_median_only = false; 
bool zero_is_valid = false;
bool seed_is_channel = false;
bool halfint_masses  = false; // find the halfling!
enum ObsAvgMode { MeanObs, LogMeanObs, MedianObs };
ObsAvgMode obs_avg_mode = MeanObs;


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

TVectorD polyFit(double x0, double y0, int npar, int n, double *xi, double *yi) {
    //std::cout << "smoothWithPolyFit(x = " << x <<", npar = " << npar << ", n = " << n << ", xi = {" << xi[0] << ", " << xi[1] << ", ...}, yi = {" << yi[0] << ", " << yi[1] << ", ...})" << std::endl;
    TMatrixDSym mat(npar);
    TVectorD    vec(npar);
    for (int j = 0; j < npar; ++j) {
        for (int j2 = j; j2 < npar; ++j2) {
            mat(j,j2) = 0;
        }
        vec(j) = 0;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < npar; ++j) {
            for (int j2 = j; j2 < npar; ++j2) {
                mat(j,j2) += std::pow(xi[i]-x0, j+j2);
            }
            vec(j) += (yi[i]-y0)*std::pow(xi[i]-x0, j);
        }
    }
    TDecompBK bk(mat);
    bk.Solve(vec);
    return vec;
}


enum BandType { Mean, Median, Quantile, Observed, Asimov, CountToys, MeanCPUTime, MeanRealTime, AdHoc, ObsQuantile };
TGraphAsymmErrors *theBand(TFile *file, int doSyst, int whichChannel, BandType type, double width=0.68) {
    if (file == 0) return 0;
    TTree *t = (TTree *) file->Get("limit");
    if (t == 0) t = (TTree *) file->Get("test"); // backwards compatibility
    if (t == 0) { std::cerr << "TFile " << file->GetName() << " does not contain the tree" << std::endl; return 0; }
    Double_t mass, limit, limitErr = 0; Float_t t_cpu, t_real; Int_t syst, iChannel, iToy, iMass; Float_t quant = -1;
    t->SetBranchAddress("mh", &mass);
    t->SetBranchAddress("limit", &limit);
    if (t->GetBranch("limitErr")) t->SetBranchAddress("limitErr", &limitErr);
    if (t->GetBranch("t_cpu") != 0) {
        t->SetBranchAddress("t_cpu", &t_cpu);
        t->SetBranchAddress("t_real", &t_real);
    }
    if (use_precomputed_quantiles) {
        if (t->GetBranch("quantileExpected") == 0) { std::cerr << "TFile " << file->GetName() << " does not have precomputed quantiles" << std::endl; return 0; }
        t->SetBranchAddress("quantileExpected", &quant);
    }
    t->SetBranchAddress("syst", &syst);
    t->SetBranchAddress((seed_is_channel ? "iSeed" : "iChannel"), &iChannel);
    t->SetBranchAddress("iToy", &iToy);

    std::map<int,std::vector<double> >  dataset;
    std::map<int,std::vector<double> >  errors;
    std::map<int,double>                obsValues;
    for (size_t i = 0, n = t->GetEntries(); i < n; ++i) {
        t->GetEntry(i);
        iMass = int(mass*100);
        //printf("%6d mh=%.1f  limit=%8.3f +/- %8.3f toy=%5d quant=% .3f\n", i, mass, limit, limitErr, iToy, quant);
        if (syst != doSyst)           continue;
        if (iChannel != whichChannel) continue;
        if      (type == Asimov)   { if (iToy != -1) continue; }
        else if (type == Observed) { if (iToy !=  0) continue; }
        else if (type == ObsQuantile && iToy == 0) { obsValues[iMass] = limit; continue; }
        else if (iToy <= 0 && !use_precomputed_quantiles) continue;
        if (limit == 0 && !zero_is_valid) continue; 
        if (type == MeanCPUTime) { 
            if (limit < 0) continue; 
            limit = t_cpu; 
        }
        if (type == MeanRealTime) { 
            if (limit < 0) continue; 
            limit = t_real; 
        }
        if (use_precomputed_quantiles) {
            if (type == CountToys)   return 0;
            if (type == Mean)        return 0;
            //std::cout << "Quantiles. What should I do " << (type == Observed ? " obs" : " exp") << std::endl;
            if (type == Observed && quant > 0) continue;
            if (type == Median) {
                if (fabs(quant - 0.5) > 0.005 && fabs(quant - (1-width)/2) > 0.005 && fabs(quant - (1+width)/2) > 0.005) {
                    //std::cout << " don't care about " << quant << std::endl;
                    continue;
                } else {
                    //std::cout << " will use " << quant << std::endl;
                }
            }
        }
        dataset[iMass].push_back(limit);
        errors[iMass].push_back(limitErr);
    }
    TGraphAsymmErrors *tge = new TGraphAsymmErrors(); 
    int ip = 0;
    for (std::map<int,std::vector<double> >::iterator it = dataset.begin(), ed = dataset.end(); it != ed; ++it) {
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
        if (use_precomputed_quantiles && type == Median) {
            if (precomputed_median_only && data.size() == 1) {
                mean = median = summer68 = winter68 = data[0];
            } else if (data.size() != 3) { 
                std::cerr << "Error for expected quantile for mass " << it->first << ": size of data is " << data.size() << std::endl; 
                continue; 
            } else {
                mean = median = data[1]; summer68 = data[0]; winter68 = data[2];
            }
        }
        double x = mean;
        switch (type) {
            case MeanCPUTime:
            case MeanRealTime:
            case Mean: x = mean; break;
            case Median: x = median; break;
            case CountToys: x = summer68 = winter68 = nd; break;
            case Asimov: // mean (in case we did it more than once), with no band
                x = summer68 = winter68 = (obs_avg_mode == mean ? mean : median);
                break;
            case Observed:
                x = mean;
                if (nd == 1) {
                    if (errors[it->first].size() == 1) {
                        summer68 = mean - errors[it->first][0];
                        winter68 = mean + errors[it->first][0];
                    } else {
                        // could happen if limitErr is not available
                        summer68 = winter68 = mean;
                    }
                } else { // if we have multiple, average and report rms (useful e.g. for MCMC)
                    switch (obs_avg_mode) {
                        case MeanObs:   x = mean; break;
                        case MedianObs: x = median; break;
                        case LogMeanObs: {
                                 x = 0;
                                 for (int j = 0; j < nd; ++j) { x += log(data[j]); }
                                  x = exp(x/nd);
                             } 
                             break;
                    }
                    double rms = 0;
                    for (int j = 0; j < nd; ++j) { rms += (x-data[j])*(x-data[j]); }
                    rms = sqrt(rms/(nd*(nd-1)));
                    summer68 = mean - rms;
                    winter68 = mean + rms;
                }
                break;
            case AdHoc:
                x = summer68 = winter68 = mean;
                break;
            case Quantile: // get the quantile equal to width, and it's uncertainty
                x = data[floor(nd*width+0.5)];
                summer68 = x - quantErr(nd, &data[0], width);
                winter68 = x + (x-summer68);
                break;
            case ObsQuantile:
                {   
                    if (obsValues.find(it->first) == obsValues.end()) continue;
                    int pass = 0, fail = 0;
                    for (int i = 0; i < nd && data[i] <= obsValues[it->first]; ++i) {
                        fail++;
                    }
                    pass = nd - fail; x = double(pass)/nd;
                    double alpha = (1.0 - .68540158589942957)/2;
                    summer68 = (pass == 0) ? 0.0 : ROOT::Math::beta_quantile(   alpha, pass,   fail+1 );
                    winter68 = (fail == 0) ? 1.0 : ROOT::Math::beta_quantile( 1-alpha, pass+1, fail   );
                    break;
                }
        } // end switch
        tge->Set(ip+1);
        tge->SetPoint(ip, it->first*0.01, x);
        tge->SetPointError(ip, 0, 0, x-summer68, winter68-x);
        ip++;
    }
    return tge;
}

TGraphAsymmErrors *theFcBelt(TFile *file, int doSyst, int whichChannel, BandType type, double width=0.68) {
    if (file == 0) return 0;
    TTree *t = (TTree *) file->Get("limit");
    if (t == 0) t = (TTree *) file->Get("test"); // backwards compatibility
    if (t == 0) { std::cerr << "TFile " << file->GetName() << " does not contain the tree" << std::endl; return 0; }
    Double_t mass, limit, limitErr = 0; Int_t syst, iChannel, iToy, iMass; Float_t quant = -1;
    t->SetBranchAddress("mh", &mass);
    t->SetBranchAddress("limit", &limit);
    t->SetBranchAddress("limitErr", &limitErr);
    t->SetBranchAddress("quantileExpected", &quant);
    t->SetBranchAddress("syst", &syst);
    t->SetBranchAddress((seed_is_channel ? "iSeed" : "iChannel"), &iChannel);
    t->SetBranchAddress("iToy", &iToy);

    TF1 fitExp("fitExp","[0]*exp([1]*(x-[2]))", 0, 1);
    TF1 fitErf("fitErf","[0]*TMath::Erfc([1]*abs(x-[2]))", 0, 1);
    std::map<int,TGraphErrors*>  dataset;
    for (size_t i = 0, n = t->GetEntries(); i < n; ++i) {
        t->GetEntry(i);
        iMass = int(mass*10);
        //printf("%6d mh=%.1f  limit=%8.3f +/- %8.3f toy=%5d quant=% .3f\n", i, mass, limit, limitErr, iToy, quant);
        if (syst != doSyst)           continue;
        if (iChannel != whichChannel) continue;
        if      (type == Asimov)   { if (iToy != -1) continue; }
        else if (type == Observed) { if (iToy !=  0) continue; }
        if (quant < 0) continue;
        TGraphErrors *& graph = dataset[iMass];
        if (graph == 0) graph = new TGraphErrors();
        int ipoint = graph->GetN(); graph->Set(ipoint+1);
        graph->SetPoint(ipoint, limit, quant); 
        graph->SetPointError(ipoint, 0, limitErr);
    }
    //std::cout << "Loaded " << dataset.size() << " masses " << std::endl;
    TGraphAsymmErrors *tge = new TGraphAsymmErrors(); int ip = 0; 
    for (std::map<int,TGraphErrors*>::iterator it = dataset.begin(), ed = dataset.end(); it != ed; ++it) {
        TGraphErrors *graph = it->second; graph->Sort();
        int n = graph->GetN(); if (n < 3) continue;
        //std::cout << "For mass " << it->first/10 << " I have " << n << " points" << std::endl;

        double blow, bhigh, bmid;

        int imax = 0; double ymax = graph->GetY()[0];
        for (int i = 0; i < n; ++i) {
            //printf(" i = %2d mH = %.1f, r = %6.3f, pval = %8.6f +/- %8.6f\n", i, it->first/10., graph->GetX()[i], graph->GetY()[i], graph->GetEY()[i]);
            if (graph->GetY()[i] > ymax) {
                imax = i; ymax = graph->GetY()[i];
            }
        }
        if (imax == 0) {
            bmid = graph->GetX()[0];
        } else if (imax == n-1) {
            bmid = graph->GetX()[n-1];
        } else {
            // ad hoc method
            double sumxmax = 0, sumwmax = 0;
            for (int i = std::max<int>(0, imax-5); i < std::max<int>(n-1,imax+5); ++i) {
                double y4 = pow(graph->GetY()[i],4);
                sumxmax += graph->GetX()[i] * y4; sumwmax += y4;
            }
            bmid = sumxmax/sumwmax;
        }

        //std::cout << "band center for " << it->first/10 << " is at " << bmid << " (imax = " << imax << ")\n" << std::endl;

        if (graph->GetY()[0] > 1-width || imax == 0) {
            blow = graph->GetX()[0];
        } else {
            int ilo = 0, ihi = 0;
            for (ilo = 1;  ilo < imax; ++ilo) {
                if (graph->GetEY()[ilo] == 0) continue;
                if (graph->GetY()[ilo]  >= 0.05*(1-width)) break;
            }
            ilo -= 1;
            for (ihi = imax; ihi > ilo+1; --ihi) {
                if (graph->GetEY()[ihi] == 0) continue;
                if (graph->GetY()[ihi] <= 3*(1-width)) break;
            }
            double xmin = graph->GetX()[ilo], xmax = graph->GetX()[ihi];
            if (ilo <= 1) xmin = 0.0001;
            fitErf.SetRange(xmin,xmax); fitErf.SetNpx(1000);
            fitErf.SetParameters(0.6,bmid,2.0/bmid);
            graph->Fit(&fitErf,"WNR EX0","",xmin,xmax);
            fitErf.SetNpx(4);
            blow = fitErf.GetX(1-width,xmin,xmax);
            if (blow <= 2*0.0001) blow = 0;
            //std::cout << width << " band low end " << it->first/10 << " is at " << blow << " (xmin = " << xmin << ", xmax = " << xmax << ")\n" << std::endl;
        }
        
        if (graph->GetY()[n-1] > 1-width || imax == n-1) {
            bhigh = graph->GetX()[n-1];
        } else if (imax == 0 && graph->GetY()[1] < 1-width) {
            double xmin = graph->GetX()[1], xmax = graph->GetX()[2];
            for (int i = 3; i <= std::max<int>(5,n); ++i) {
                 if (graph->GetY()[i] < 0.5*(1-width)) break;
                 xmax = graph->GetX()[i];
            }
            fitExp.SetRange(xmin,xmax); fitExp.SetNpx(1000);
            fitExp.SetParameters(1-width, -2.0/(xmax-xmin), 0.5*(xmax-xmin));
            fitExp.FixParameter(0,1-width);
            graph->Fit(&fitExp,"WNR EX0","",xmin,xmax);
            bhigh = fitExp.GetParameter(2);
            if (bhigh < graph->GetX()[0]) {
                bhigh = graph->GetX()[0] + ((1-width)-graph->GetY()[0])*(graph->GetX()[1]-graph->GetX()[0])/(graph->GetY()[1]-graph->GetY()[0]);
                //std::cout << width << " band high end forces stupid linear interpolation" << std::endl;
            }
            //std::cout << width << " band high end " << it->first/10 << " is at " << bhigh << " (xmin = " << xmin << ", xmax = " << xmax << ")\n" << std::endl;
        } else {
            int ilo = 0, ihi = 0;
            for (ilo = imax+1;  ilo < n-2; ++ilo) {
                if (graph->GetEY()[ilo] == 0) continue;
                if (graph->GetY()[ilo]  <= 3*(1-width)) break;
            }
            if (ilo > 0 && graph->GetEY()[ilo-1] != 0) ilo--;
            for (ihi = ilo+1; ihi < n; ++ihi) {
                if (graph->GetEY()[ihi] == 0) { ihi--; break; }
                if (graph->GetY()[ihi] >= 0.05*(1-width)) break;
            }
            if (ihi - ilo <= 1) { 
                double xmin = graph->GetX()[ilo], xmax = graph->GetX()[ihi];
                bhigh = 0.5*(xmin+xmax);
                //std::cout << width << " band high end " << it->first/10 << " is " << bhigh << " (xmin = " << xmin << ", xmax = " << xmax << ", no fit)\n" << std::endl;
            } else {
                double xmin = graph->GetX()[ilo], xmax = graph->GetX()[ihi];
                fitExp.SetRange(xmin,xmax); fitExp.SetNpx(1000);
                fitExp.SetParameters(1-width, -2.0/(xmax-xmin), 0.5*(xmax-xmin));
                fitExp.FixParameter(0,1-width);
                graph->Fit(&fitExp,"WNR EX0","",xmin,xmax);
                bhigh = fitExp.GetParameter(2);
                //std::cout << width << " band high end " << it->first/10 << " is at " << bhigh << " (xmin = " << xmin << ", xmax = " << xmax << ")\n" << std::endl;
            }
        }

        tge->Set(ip+1);
        tge->SetPoint(ip, it->first*0.1, bmid);
        tge->SetPointError(ip, 0, 0, bmid-blow, bhigh-bmid);
        ip++;

        continue;
        delete graph;
    }
    return tge;
}

void theBand() {}
bool do_bands_95 = true;
void makeBand(TDirectory *bands, TString name, TFile *file, int doSyst, int whichChannel, BandType type) {
    TString suffix = "";
    switch (type) {
        case Asimov:    suffix = "_asimov"; break;
        case Observed:  suffix = "_obs"; break;
        case Mean:      suffix = "_mean"; break;
        case Median:    suffix = "_median"; break;
        case CountToys: suffix = "_ntoys"; break;
        case MeanCPUTime: suffix = "_cputime"; break;
        case MeanRealTime: suffix = "_realtime"; break;
        case AdHoc:       suffix = ""; break;
        case Quantile:    suffix = ""; break;
        case ObsQuantile:    suffix = "_qobs"; break;
    }
    if (!doSyst && (type != AdHoc)) suffix = "_nosyst"+suffix;
    if (type == Median || type == Mean) {
        TGraph *band68 = theBand(file, doSyst, whichChannel, type, 0.68);
        TGraph *band95 = do_bands_95 ? theBand(file, doSyst, whichChannel, type, 0.95) : 0; 
        if (band68 != 0 && band68->GetN() > 0) {
            band68->SetName(name+suffix);
            bands->WriteTObject(band68, name+suffix);
            if (do_bands_95) {
                band95->SetName(name+suffix+"_95");
                bands->WriteTObject(band95, name+suffix+"_95");
            }
        } else {
            std::cout << "Band " << name+suffix << " missing" << std::endl;
        }
    } else {
        TGraph *band = theBand(file, doSyst, whichChannel, type);
        if (band != 0 && band->GetN() > 0) {
            band->SetName(name+suffix);
            bands->WriteTObject(band, name+suffix);
            //std::cout << "Band " << name+suffix << " found (" << band->GetN() << " points)" << std::endl;
        } else {
            std::cout << "Band " << name+suffix << " missing" << std::endl;
        }
    }
}
void makeBand(TDirectory *bands, TString name, TString filename, int doSyst, int whichChannel, BandType type) {
    TFile *in = TFile::Open(filename);
    if (in == 0) { std::cerr << "Filename " << filename << " missing" << std::endl; return; }
    makeBand(bands, name, in, doSyst, whichChannel, type);
}
void makeLine(TDirectory *bands, TString name, TString filename,  int doSyst, int whichChannel) {
    TFile *in = TFile::Open(filename);
    if (in == 0) { std::cerr << "Filename '" << filename << "' missing" << std::endl; return; }
    makeBand(bands, name, in, doSyst, whichChannel, AdHoc);
    in->Close();
}

bool do_bands_nosyst = true;
bool do_bands_mean = true;
bool do_bands_median = true;
bool do_bands_ntoys = true;
bool do_bands_asimov = true;
bool do_bands_cputime = false;
bool do_bands_realtime = false;
void makeBands(TDirectory *bands, TString name, TString filename, int channel=0, bool quantiles=false) {
    TFile *in = TFile::Open(filename);
    if (in == 0) { std::cerr << "Filename " << filename << " missing" << std::endl; return; }
    for (int s = do_bands_nosyst ? 0 : 1; s <= 1; ++s) {
        if (do_bands_mean) makeBand(bands, name, in, s, channel, Mean);
        makeBand(bands, name, in, s, channel, Median);
        makeBand(bands, name, in, s, channel, Observed);
        if (do_bands_ntoys)  makeBand(bands, name, in, s, channel, CountToys);
        if (do_bands_asimov) makeBand(bands, name, in, s, channel, Asimov);
        if (do_bands_cputime)  makeBand(bands, name, in, s, channel, MeanCPUTime);
        if (do_bands_realtime) makeBand(bands, name, in, s, channel, MeanRealTime);
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

int findBin(TGraph *g, double x, double tolerance) {
    if (g == 0) return -1;
    for (int i = 0; i < g->GetN(); ++i) {
        double xi = g->GetX()[i];
        if (fabs(xi -  x) < tolerance) {
            return i;
        }
    }
    return -1;
}
int findBin(TGraphAsymmErrors *g, double x) {
    if (g == 0) return -1;
    for (int i = 0; i < g->GetN(); ++i) {
        double xi = g->GetX()[i];
        if ((xi - g->GetErrorXlow(i) <= x) && (x <= xi + g->GetErrorXhigh(i))) {
            return i;
        }
    }
    return -1;
}

void significanceToPVal(TDirectory *bands, TString inName, TString outName) {
    TGraphAsymmErrors *b1 = (TGraphAsymmErrors *) bands->Get(inName);
    if (b1 == 0 || b1->GetN() == 0) return;
    int n = b1->GetN();
    TGraphAsymmErrors *b2 = new TGraphAsymmErrors(n);
    for (int i = 0; i < n; ++i) {
        double x = b1->GetX()[i], s = b1->GetY()[i];
        double slo = s - b1->GetErrorYlow(i), shi = s + b1->GetErrorYhigh(i);
        double pval = ROOT::Math::normal_cdf_c(s);
        double phi  = ROOT::Math::normal_cdf_c(slo);
        double plo  = ROOT::Math::normal_cdf_c(shi);
        b2->SetPoint(i, x, pval);
        b2->SetPointError(i, b1->GetErrorXlow(i), b1->GetErrorXhigh(i), pval - plo, phi - pval);
    }
    b2->SetName(outName);
    bands->WriteTObject(b2, outName);
}
void pvalToSignificance(TDirectory *bands, TString inName, TString outName) {
    TGraphAsymmErrors *b1 = (TGraphAsymmErrors *) bands->Get(inName);
    if (b1 == 0 || b1->GetN() == 0) return;
    int n = b1->GetN();
    TGraphAsymmErrors *b2 = new TGraphAsymmErrors(n);
    for (int i = 0; i < n; ++i) {
        double x = b1->GetX()[i], s = b1->GetY()[i];
        double slo = s - b1->GetErrorYlow(i), shi = s + b1->GetErrorYhigh(i);
        double pval = ROOT::Math::normal_quantile_c(s,   1.0);
        double phi  = ROOT::Math::normal_quantile_c(slo, 1.0);
        double plo  = ROOT::Math::normal_quantile_c(shi, 1.0);
        b2->SetPoint(i, x, pval);
        b2->SetPointError(i, b1->GetErrorXlow(i), b1->GetErrorXhigh(i), pval - plo, phi - pval);
    }
    b2->SetName(outName);
    bands->WriteTObject(b2, outName);
}

void testStatToPVal(TDirectory *bands, TString inName, TString outName) {
    TGraphAsymmErrors *b1 = (TGraphAsymmErrors *) bands->Get(inName);
    if (b1 == 0 || b1->GetN() == 0) return;
    int n = b1->GetN();
    TGraphAsymmErrors *b2 = new TGraphAsymmErrors(n);
    for (int i = 0; i < n; ++i) {
        double x = b1->GetX()[i], s = b1->GetY()[i];
        double slo = s - b1->GetErrorYlow(i), shi = s + b1->GetErrorYhigh(i);
        s   = (s   > 0 ? sqrt(s  ) : 0);
        slo = (slo > 0 ? sqrt(slo) : 0);
        shi = (shi > 0 ? sqrt(shi) : 0);
        double pval = ROOT::Math::normal_cdf_c(s);
        double phi  = ROOT::Math::normal_cdf_c(slo);
        double plo  = ROOT::Math::normal_cdf_c(shi);
        b2->SetPoint(i, x, pval);
        b2->SetPointError(i, b1->GetErrorXlow(i), b1->GetErrorXhigh(i), pval - plo, phi - pval);

    }
    b2->SetName(outName);
    bands->WriteTObject(b2, outName);
}

void significanceToPVals(TDirectory *bands, TString inName, TString outName) {
    significanceToPVal(bands, inName+"_obs",   outName+"_obs");
    significanceToPVal(bands, inName+"_mean",   outName+"_mean");
    significanceToPVal(bands, inName+"_median", outName+"_median");
    significanceToPVal(bands, inName+"_mean_95",   outName+"_mean_95");
    significanceToPVal(bands, inName+"_median_95", outName+"_median_95");
    significanceToPVal(bands, inName+"_asimov",    outName+"_asimov");

    significanceToPVal(bands, inName+"_nosyst_obs",   outName+"_nosyst_obs");
    significanceToPVal(bands, inName+"_nosyst_mean",   outName+"_nosyst_mean");
    significanceToPVal(bands, inName+"_nosyst_median", outName+"_nosyst_median");
    significanceToPVal(bands, inName+"_nosyst_mean_95",   outName+"_nosyst_mean_95");
    significanceToPVal(bands, inName+"_nosyst_asimov",    outName+"_nosyst_asimov");
    significanceToPVal(bands, inName+"_nosyst_ntoys",     outName+"_nosyst_ntoys");
}
void testStatToPVals(TDirectory *bands, TString inName, TString outName) {
    testStatToPVal(bands, inName+"_obs",   outName+"_obs");
    testStatToPVal(bands, inName+"_mean",   outName+"_mean");
    testStatToPVal(bands, inName+"_median", outName+"_median");
    testStatToPVal(bands, inName+"_mean_95",   outName+"_mean_95");
    testStatToPVal(bands, inName+"_median_95", outName+"_median_95");
    testStatToPVal(bands, inName+"_asimov",    outName+"_asimov");

    testStatToPVal(bands, inName+"_nosyst_obs",   outName+"_nosyst_obs");
    testStatToPVal(bands, inName+"_nosyst_mean",   outName+"_nosyst_mean");
    testStatToPVal(bands, inName+"_nosyst_median", outName+"_nosyst_median");
    testStatToPVal(bands, inName+"_nosyst_mean_95",   outName+"_nosyst_mean_95");
    testStatToPVal(bands, inName+"_nosyst_asimov",    outName+"_nosyst_asimov");
    testStatToPVal(bands, inName+"_nosyst_ntoys",     outName+"_nosyst_ntoys");
}


void cutBand(TDirectory *bands, TString inName, TString outName, float mMin, float mMax) {
    TGraphAsymmErrors *b1 = (TGraphAsymmErrors *) bands->Get(inName);
    if (b1 == 0 || b1->GetN() == 0) return;
    TGraphAsymmErrors *b2 = new TGraphAsymmErrors();
    int n = b1->GetN(), m = 0;
    for (int i = 0; i < n; ++i) {
        if (mMin <= b1->GetX()[i] && b1->GetX()[i] <= mMax) {
            b2->Set(m+1);
            b2->SetPoint(m, b1->GetX()[i], b1->GetY()[i]);
            b2->SetPointError(m, b1->GetErrorXlow(i), b1->GetErrorXhigh(i),
                                 b1->GetErrorYlow(i), b1->GetErrorYhigh(i));
            m++;
        }
    }
    b2->SetName(outName);
    bands->WriteTObject(b2, outName);
    //bands->Add(b2);
}

void cutBands(TDirectory *bands, TString inName, TString outName, float mMin, float mMax) {
    cutBand(bands, inName+"_obs",   outName+"_obs",    mMin, mMax);
    cutBand(bands, inName+"_mean",   outName+"_mean",    mMin, mMax);
    cutBand(bands, inName+"_median", outName+"_median",  mMin, mMax);
    cutBand(bands, inName+"_mean_95",   outName+"_mean_95",    mMin, mMax);
    cutBand(bands, inName+"_median_95", outName+"_median_95",  mMin, mMax);
    cutBand(bands, inName+"_asimov",    outName+"_asimov",     mMin, mMax);
    cutBand(bands, inName+"_ntoys",     outName+"_ntoys",      mMin, mMax);

    cutBand(bands, inName+"_nosyst_obs",   outName+"_nosyst_obs",    mMin, mMax);
    cutBand(bands, inName+"_nosyst_mean",   outName+"_nosyst_mean",    mMin, mMax);
    cutBand(bands, inName+"_nosyst_median", outName+"_nosyst_median",  mMin, mMax);
    cutBand(bands, inName+"_nosyst_mean_95",   outName+"_nosyst_mean_95",    mMin, mMax);
    cutBand(bands, inName+"_nosyst_asimov",    outName+"_nosyst_asimov",     mMin, mMax);
    cutBand(bands, inName+"_nosyst_ntoys",     outName+"_nosyst_ntoys",      mMin, mMax);
}
void cutFcBands(TDirectory *bands, TString inName, TString outName, float mMin, float mMax, int npostfix, char **postfixes) {
    for (int i = 0; i < npostfix; ++i) {
        cutBand(bands, inName+"_"+postfixes[i],   outName+"_"+postfixes[i],  mMin, mMax);
    } 
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


void mergeBand(TDirectory *in, TString band1, TString band2, TString comb) {
    TGraphAsymmErrors *b1 = (TGraphAsymmErrors *) in->Get(band1);
    TGraphAsymmErrors *b2 = (TGraphAsymmErrors *) in->Get(band2);
    if (b1 == 0 || b1->GetN() == 0) return;
    if (b2 == 0 || b2->GetN() == 0) return;
    int n = b1->GetN(), m = b2->GetN();
    TGraphAsymmErrors *bc = new TGraphAsymmErrors(n);
    bc->SetName(comb); 
    int k = 0;
    for (int i = 0; i < n; ++i, ++k) {
        bc->SetPoint(k, b1->GetX()[i], b1->GetY()[i]);
        bc->SetPointError(k, b1->GetErrorXlow(i), b1->GetErrorXhigh(i), 
                             b1->GetErrorYlow(i), b1->GetErrorYhigh(i));
    }
    for (int i = 0; i < m; ++i) {
        if (findBin(b1, b2->GetX()[i], 0.001) != -1) continue;
        bc->Set(k);
        bc->SetPoint(k, b2->GetX()[i], b2->GetY()[i]);
        bc->SetPointError(k, b2->GetErrorXlow(i), b2->GetErrorXhigh(i), 
                             b2->GetErrorYlow(i), b2->GetErrorYhigh(i));
        k++;
    }
    bc->Sort();
    bc->SetName(comb);
    in->WriteTObject(bc, comb, "Overwrite");
}
void mergeBands(TDirectory *in, TString band1, TString band2, TString comb) {
    mergeBand(in, band1+"_obs",   band2+"_obs",   comb+"_obs");
    mergeBand(in, band1+"_mean",   band2+"_mean",   comb+"_mean");
    mergeBand(in, band1+"_median", band2+"_median", comb+"_median");
    mergeBand(in, band1+"_mean_95",   band2+"_mean_95",   comb+"_mean_95");
    mergeBand(in, band1+"_median_95", band2+"_median_95", comb+"_median_95");
    mergeBand(in, band1+"_asimov",    band2+"_asimov",    comb+"_asimov");
    mergeBand(in, band1+"_ntoys",    band2+"_ntoys",    comb+"_ntoys");

    mergeBand(in, band1+"_nosyst_mean",   band2+"_nosyst_mean",   comb+"_nosyst_mean");
    mergeBand(in, band1+"_nosyst_median", band2+"_nosyst_median", comb+"_nosyst_median");
    mergeBand(in, band1+"_nosyst_mean_95",   band2+"_nosyst_mean_95",   comb+"_nosyst_mean_95");
    mergeBand(in, band1+"_nosyst_median_95", band2+"_nosyst_median_95", comb+"_nosyst_median_95");
    mergeBand(in, band1+"_nosyst_asimov",    band2+"_nosyst_asimov",    comb+"_nosyst_asimov");
    mergeBand(in, band1+"_nosyst_ntoys",    band2+"_nosyst_ntoys",    comb+"_nosyst_ntoys");
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
    pasteBand(in, band1+"_obs",   band2+"_obs",   comb+"_obs");
    pasteBand(in, band1+"_mean",   band2+"_mean",   comb+"_mean");
    pasteBand(in, band1+"_median", band2+"_median", comb+"_median");
    pasteBand(in, band1+"_mean_95",   band2+"_mean_95",   comb+"_mean_95");
    pasteBand(in, band1+"_median_95", band2+"_median_95", comb+"_median_95");
    pasteBand(in, band1+"_asimov",    band2+"_asimov",    comb+"_asimov");
    pasteBand(in, band1+"_ntoys",    band2+"_ntoys",    comb+"_ntoys");

    pasteBand(in, band1+"_nosyst_obs",   band2+"_nosyst_obs",   comb+"_nosyst_obs");
    pasteBand(in, band1+"_nosyst_mean",   band2+"_nosyst_mean",   comb+"_nosyst_mean");
    pasteBand(in, band1+"_nosyst_median", band2+"_nosyst_median", comb+"_nosyst_median");
    pasteBand(in, band1+"_nosyst_mean_95",   band2+"_nosyst_mean_95",   comb+"_nosyst_mean_95");
    pasteBand(in, band1+"_nosyst_median_95", band2+"_nosyst_median_95", comb+"_nosyst_median_95");
    pasteBand(in, band1+"_nosyst_asimov",    band2+"_nosyst_asimov",    comb+"_nosyst_asimov");
    pasteBand(in, band1+"_nosyst_ntoys",    band2+"_nosyst_ntoys",    comb+"_nosyst_ntoys");
}
void pasteFcBands(TDirectory *bands, TString band1, TString band2, TString comb, int npostfix, char **postfixes) {
    for (int i = 0; i < npostfix; ++i) {
        pasteBand(bands, band1+"_"+postfixes[i],   band2+"_"+postfixes[i],   comb+"_"+postfixes[i]);
    } 
}


void stripPoint(TGraph *band, float m) {
    for (int i = 0, n = band->GetN(); i < n; ++i) {
        if (float(band->GetX()[i]) == m) {
            band->RemovePoint(i);
            return;
        }
    }
    if ((band->GetN() > 0) &&
        (band->GetX()[0] <= m) &&
        (band->GetX()[band->GetN()-1] >= m)) {
    }
}
void stripBand(TDirectory *in, TString band1, float m1, float m2=0, float m3=0, float m4=0, float m5=0) {
    TGraphAsymmErrors *band = (TGraphAsymmErrors *) in->Get(band1);
    if (band == 0 || band->GetN() == 0) return; 
    if (m1) stripPoint(band,m1);
    if (m2) stripPoint(band,m2);
    if (m3) stripPoint(band,m3);
    if (m4) stripPoint(band,m4);
    if (m5) stripPoint(band,m5);
    in->WriteTObject(band, band->GetName(), "Overwrite");
}

void stripBands(TDirectory *in, TString band,  float m1, float m2=0, float m3=0, float m4=0, float m5=0) {
    stripBand(in, band+"_obs",       m1,m2,m3,m4,m5);
    stripBand(in, band+"_mean",      m1,m2,m3,m4,m5);
    stripBand(in, band+"_median",    m1,m2,m3,m4,m5);
    stripBand(in, band+"_mean_95",   m1,m2,m3,m4,m5);
    stripBand(in, band+"_median_95", m1,m2,m3,m4,m5);
    stripBand(in, band+"_asimov",    m1,m2,m3,m4,m5);
    stripBand(in, band+"_nosyst_obs",       m1,m2,m3,m4,m5);
    stripBand(in, band+"_nosyst_mean",      m1,m2,m3,m4,m5);
    stripBand(in, band+"_nosyst_median",    m1,m2,m3,m4,m5);
    stripBand(in, band+"_nosyst_mean_95",   m1,m2,m3,m4,m5);
    stripBand(in, band+"_nosyst_median_95", m1,m2,m3,m4,m5);
    stripBand(in, band+"_nosyst_asimov",    m1,m2,m3,m4,m5);
}

void copyPoint(TGraphAsymmErrors *from, float m, TGraphAsymmErrors *to, int idx=-1) {
    int j = findBin(from, m);
    if (j == -1) return;
    if (idx == -1) { idx = to->GetN(); to->Set(idx+1); }
    to->SetPoint(idx, from->GetX()[j], from->GetY()[j]);
    to->SetPointError(idx, from->GetErrorXlow(j), from->GetErrorXhigh(j), from->GetErrorYlow(j), from->GetErrorYhigh(j));
}
void selectedPointsBand(TDirectory *in, TString band1, TString band2, float m1, float m2=0, float m3=0, float m4=0, float m5=0, float m6=0, float m7=0){
    TGraphAsymmErrors *band = (TGraphAsymmErrors *) in->Get(band1);
    if (band == 0 || band->GetN() == 0) return; 
    TGraphAsymmErrors *ret = new TGraphAsymmErrors();
    copyPoint(band, m1, ret);
    if (m2) copyPoint(band, m2, ret);
    if (m3) copyPoint(band, m3, ret);
    if (m4) copyPoint(band, m4, ret);
    if (m5) copyPoint(band, m5, ret);
    if (m6) copyPoint(band, m6, ret);
    if (m7) copyPoint(band, m7, ret);
    ret->SetName(band2);
    in->WriteTObject(ret, ret->GetName(), "Overwrite");
}

void selectedPointsBands(TDirectory *in, TString bandIn, TString bandOut,  float m1, float m2=0, float m3=0, float m4=0, float m5=0, float m6=0, float m7=0) {
    selectedPointsBand(in, bandIn+"_obs",       bandOut+"_obs",       m1,m2,m3,m4,m5,m6,m7);
    selectedPointsBand(in, bandIn+"_mean",      bandOut+"_mean",      m1,m2,m3,m4,m5,m6,m7);
    selectedPointsBand(in, bandIn+"_median",    bandOut+"_median",    m1,m2,m3,m4,m5,m6,m7);
    selectedPointsBand(in, bandIn+"_mean_95",   bandOut+"_mean_95",   m1,m2,m3,m4,m5,m6,m7);
    selectedPointsBand(in, bandIn+"_median_95", bandOut+"_median_95", m1,m2,m3,m4,m5,m6,m7);
    selectedPointsBand(in, bandIn+"_asimov",    bandOut+"_asimov",    m1,m2,m3,m4,m5,m6,m7);
    selectedPointsBand(in, bandIn+"_nosyst_obs",       bandOut+"_nosyst_obs",       m1,m2,m3,m4,m5,m6,m7);
    selectedPointsBand(in, bandIn+"_nosyst_mean",      bandOut+"_nosyst_mean",      m1,m2,m3,m4,m5,m6,m7);
    selectedPointsBand(in, bandIn+"_nosyst_median",    bandOut+"_nosyst_median",    m1,m2,m3,m4,m5,m6,m7);
    selectedPointsBand(in, bandIn+"_nosyst_mean_95",   bandOut+"_nosyst_mean_95",   m1,m2,m3,m4,m5,m6,m7);
    selectedPointsBand(in, bandIn+"_nosyst_median_95", bandOut+"_nosyst_median_95", m1,m2,m3,m4,m5,m6,m7);
    selectedPointsBand(in, bandIn+"_nosyst_asimov",    bandOut+"_nosyst_asimov",    m1,m2,m3,m4,m5,m6,m7);
}


void printLine(TDirectory *bands, TString who, FILE *fout, TString header="value") {
    TGraphAsymmErrors *mean = (TGraphAsymmErrors*) bands->Get(who);
    if (mean == 0) { std::cerr << "MISSING " << who << std::endl; return; }
    fprintf(fout, "%4s \t %7s\n", "mass",  header.Data());
    fprintf(fout,  "%5s\t %7s\n", "-----", "-----");
    TString fmtstring = TString() +
                        (halfint_masses ? "%5.1f" : "%4.0f ") +
                        "\t " +
                        (who.Contains("pval")  || who.Contains("smcls")  ? "%7.5f" : "%7.3f") +
                        "\n";
    for (int i = 0, n = mean->GetN(); i < n; ++i) {
        fprintf(fout, fmtstring.Data(),  mean->GetX()[i], mean->GetY()[i]);  
    }
}
void printLine(TDirectory *bands, TString who, TString fileName, TString header="value") {
    TGraph *mean = (TGraph*) bands->Get(who);
    if (mean == 0) { std::cerr << "MISSING " << who << std::endl; return; }
    FILE *fout = fopen(fileName.Data(), "w");
    printLine(bands,who,fout,header);
    fclose(fout);
}

void printLineErr(TDirectory *bands, TString who, FILE *fout, TString header="value") {
    TGraphAsymmErrors *mean = (TGraphAsymmErrors*) bands->Get(who);
    if (mean == 0) { std::cerr << "MISSING " << who << std::endl; return; }
    fprintf(fout, "%4s \t %7s +/- %6s\n", "mass",  header.Data()," error");
    fprintf(fout,  "%5s\t %7s-----%6s-\n", "-----", " ------","------");
    TString fmtstring = TString() +
                        (halfint_masses ? "%5.1f" : "%4.0f ") + 
                        "\t " +
                        (who.Contains("pval")  || who.Contains("smcls")  ? "%7.5f +/- %7.5f" : "%7.3f +/- %6.3f") + 
                        "\n";
    for (int i = 0, n = mean->GetN(); i < n; ++i) {
        fprintf(fout, fmtstring.Data(),  
            mean->GetX()[i], 
            mean->GetY()[i], 
            TMath::Max(mean->GetErrorYlow(i),mean->GetErrorYhigh(i)));  
    }
}
void printLineErr(TDirectory *bands, TString who, TString fileName, TString header="value") {
    TGraph *mean = (TGraph*) bands->Get(who);
    if (mean == 0) { std::cerr << "MISSING " << who << std::endl; return; }
    FILE *fout = fopen(fileName.Data(), "w");
    if (fout == 0)  { std::cerr << "CANNOT WRITE TO " << fileName << std::endl; return; }
    printLineErr(bands,who,fout,header);
    fclose(fout);
}

void printLineAErr(TDirectory *bands, TString who, FILE *fout, TString header="value") {
    TGraphAsymmErrors *mean = (TGraphAsymmErrors*) bands->Get(who);
    if (mean == 0) { std::cerr << "MISSING " << who << std::endl; return; }
    fprintf(fout, "%4s \t %7s  -%6s   +%6s\n",  "mass",  header.Data(),"error"," error");
    TString fmtstring = TString() +
                        (halfint_masses ? "%5.1f" : "%4.0f ") + 
                        "\t " +
                        (who.Contains("pval") || who.Contains("smcls")   ? "%7.5f  -%7.5f / +%7.5f" : "%7.3f  -%6.3f / +%6.3f") + 
                        "\n";
   fprintf(fout,  "%5s\t %7s------%6s----%6s-\n", "-----", " ------","------","------");
    for (int i = 0, n = mean->GetN(); i < n; ++i) {
        fprintf(fout, fmtstring.Data(),  
            mean->GetX()[i], 
            mean->GetY()[i], 
            mean->GetErrorYlow(i),mean->GetErrorYhigh(i));  
    }
}
void printLineAErr(TDirectory *bands, TString who, TString fileName, TString header="value") {
    TGraph *mean = (TGraph*) bands->Get(who);
    if (mean == 0) { std::cerr << "MISSING " << who << std::endl; return; }
    FILE *fout = fopen(fileName.Data(), "w");
    printLineAErr(bands,who,fout,header);
    fclose(fout);
}



void printBand(TDirectory *bands, TString who, FILE *fout, bool mean=false) {
    TGraphAsymmErrors *obs    = (TGraphAsymmErrors*) bands->Get(who+"_obs");
    TGraphAsymmErrors *mean68 = (TGraphAsymmErrors*) bands->Get(who+(mean?"_mean":"_median"));
    TGraphAsymmErrors *mean95 = (TGraphAsymmErrors*) bands->Get(who+(mean?"_mean_95":"_median_95"));
    if (mean68 == 0 && obs == 0) { std::cerr << "MISSING " << who << "_mean and " << who << "_obs" << std::endl; return; }
    if (mean68 == 0) { printLineErr(bands, who+"_obs", fout); return; }
    fprintf(fout, "%4s \t %8s  %8s  %8s  %8s  %8s  %8s\n", "mass", " obs ", "-95%", "-68%", (mean ? "mean" : "median"), "+68%", "+95%");
    fprintf(fout,  "%5s\t %8s  %8s  %8s  %8s  %8s  %8s\n", "-----","-----",  "-----", "-----", "-----", "-----", "-----");
    TString fmtstring = TString() +
                        (halfint_masses ? "%5.1f" : "%4.0f ") + 
                        "\t " +
                        (who.Contains("pval") || who.Contains("smcls")  ? "%8.6f  %8.6f  %8.6f  %8.6f  %8.6f  %8.6f" : "%8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f") + 
                        "\n";
    for (int i = 0, n = mean68->GetN(); i < n; ++i) {
        int j  = (obs    ? findBin(obs,    mean68->GetX()[i]) : -1);
        int j2 = (mean95 ? findBin(mean95, mean68->GetX()[i]) : -1);
        fprintf(fout, fmtstring.Data() , 
            mean68->GetX()[i],  
            j == -1 ? NAN : obs->GetY()[j],
            j2 == -1 ? NAN : mean95->GetY()[j2]-mean95->GetErrorYlow(j2), 
            mean68->GetY()[i]-mean68->GetErrorYlow(i), 
            mean68->GetY()[i],
            mean68->GetY()[i]+mean68->GetErrorYhigh(i),
            j2 == -1 ? NAN : mean95->GetY()[j2]+mean95->GetErrorYhigh(j2)
        );
    }
}
void printFcBand(TDirectory *bands, TString who, FILE *fout, int npostfix, char **postfixes) {
    TGraphAsymmErrors *bs[99]; char *names[99]; int nbands = 0;
    for (int i = 0; i < npostfix; ++i) {
        bs[nbands] = (TGraphAsymmErrors*) bands->Get(who+"_"+postfixes[i]);
        if (bs[nbands] != 0) { names[nbands] = postfixes[i]; nbands++; }
    }
    if (nbands == 0) return;
    printf("Found %d bands\n", nbands);

    fprintf(fout, "%4s \t ", "mass");
    for (int i = 0; i < nbands; ++i) fprintf(fout, " -%-5s  ", names[nbands-i-1]);
    fprintf(fout, "  %8s  ", "  mid.  ");
    for (int i = 0; i < nbands; ++i) fprintf(fout, "   +%-5s", names[i]);
    fprintf(fout, "\n");

    for (int i = 0, n = bs[0]->GetN(); i < n; ++i) {
        double xi = bs[0]->GetX()[i]; 
        fprintf(fout, (halfint_masses ? "%5.1f\t" : "%4.0f \t"), xi);

        for (int j = nbands-1; j >= 0; --j) {
            int ij = findBin(bs[j], xi);
            double y = (ij == -1 ? NAN : bs[j]->GetY()[ij] - bs[j]->GetErrorYlow(ij));
            fprintf(fout, "%7.5f  ", y);
        }

        fprintf(fout, "  %7.5f  ", bs[0]->GetY()[i]);

        for (int j = 0; j < nbands; ++j) {
            int ij = findBin(bs[j], xi);
            double y = (ij == -1 ? NAN : bs[j]->GetY()[ij] + bs[j]->GetErrorYhigh(ij));
            fprintf(fout, "  %7.5f", y);
        }

        fprintf(fout, "\n");
    }
}
void printFcBand(TDirectory *bands, TString who, TString fileName, int npostfix, char **postfixes) {
    TGraph *first = (TGraph*) bands->Get(who+"_"+postfixes[0]); 
    if (first == 0) { std::cerr << "MISSING " << who << "_" << postfixes[0] << std::endl; return; }
    FILE *fout = fopen(fileName.Data(), "w");
    printFcBand(bands, who, fout, npostfix, postfixes);
    fclose(fout);
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
void printBand(TDirectory *bands, TString who, TString fileName, bool mean=false) {
    TGraph *mean68 = (TGraph*) bands->Get(who+(mean?"_mean":"_median"));
    TGraphAsymmErrors *obs  = (TGraphAsymmErrors*) bands->Get(who+"_obs");
    if (mean68 == 0 && obs == 0) { 
        std::cerr << "MISSING " << who << "_mean and " << who << "_obs" << std::endl; 
        return; 
    }
    FILE *fout = fopen(fileName.Data(), "w");
    printBand(bands,who,fout,mean);
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
    TGraphAsymmErrors *in68  = new TGraphAsymmErrors();  in68->SetName(name+"_median");
    TGraphAsymmErrors *in95  = new TGraphAsymmErrors();  in95->SetName(name+"_median_95");
    float mH, yObs, yLL, yLo, y, yHi, yHH; 
    char buff[1025];
    do {
        int c = fgetc(in);
        if (c == 'm' || c == '-') {
            fgets(buff,1024,in);
        } else {
            ungetc(c,in);
            break;
        }
    } while(true);
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

void importLandS(TDirectory *bands, TString name, TFile *file, bool doObserved=true, bool doExpected=true) {
    if (file == 0) return ;
    TTree *t = (TTree *) file->Get("T");
    if (t == 0) { std::cerr << "TFile " << file->GetName() << " does not contain the tree" << std::endl; return; }
    bool isML = (name.Index("ml") == 0), isPVal = (name.Index("pval") == 0);
    Double_t mass, limit, limitErr, rmedian, rm1s, rp1s, rm2s, rp2s;

    t->SetBranchAddress("mH", &mass);
    t->SetBranchAddress("rmedian", &rmedian);
    t->SetBranchAddress("rm1s", &rm1s);
    t->SetBranchAddress("rm2s", &rm2s);
    t->SetBranchAddress("rp2s", &rp2s);
    t->SetBranchAddress("rp1s", &rp1s);
    TString what = "limit";
    if (isPVal) what = "pvalue";
    if (isML)   what = "rmean";
    std::cout << "For " << name << " will read " << what.Data() << std::endl;
    t->SetBranchAddress(what.Data(), &limit);
    t->SetBranchAddress("limitErr", &limitErr);
    TGraphAsymmErrors *obs       = new TGraphAsymmErrors(); obs->SetName(name+"_obs");             int nobs = 0;
    TGraphAsymmErrors *median    = new TGraphAsymmErrors(); median->SetName(name+"_median");       int nmedian = 0;
    TGraphAsymmErrors *median_95 = new TGraphAsymmErrors(); median_95->SetName(name+"_median_95"); int nmedian_95 = 0;
    for (size_t i = 0, n = t->GetEntries(); i < n; ++i) {
        t->GetEntry(i);
        if (doObserved) {
            if (isML) {
                obs->Set(nobs+1);
                obs->SetPoint(nobs, mass, limit);
                obs->SetPointError(nobs, 0, 0, limit-rm1s, rp1s-limit);
                nobs++; 
            } else if (limit != 0) {
                obs->Set(nobs+1);
                obs->SetPoint(nobs, mass, limit);
                obs->SetPointError(nobs, 0, 0, limitErr, limitErr);
                nobs++; 
            }
        }
        if (!isML && doExpected) {
            if (isPVal) { 
                if (limit != 0) {
                    median->Set(nmedian+1);
                    median->SetPoint(nmedian, mass, limit);
                    median->SetPointError(nmedian, 0, 0, 0, 0);
                    nmedian++; 
                } 
            } else {
                if (limit != 0) {
                    median->Set(nmedian+1);
                    median->SetPoint(nmedian, mass, rmedian);
                    if (rm1s != 0 && rp1s != 0) {
                        median->SetPointError(nmedian, 0, 0, rmedian - rm1s, rp1s - rmedian);
                    } else {
                        median->SetPointError(nmedian, 0, 0, 0, 0);
                    }
                    nmedian++; 
                }
                if (limit != 0 && rm2s != 0 && rp2s != 0) {
                    median_95->Set(nmedian_95+1);
                    median_95->SetPoint(nmedian_95, mass, rmedian);
                    median_95->SetPointError(nmedian_95, 0, 0, rmedian - rm2s, rp2s - rmedian);
                    nmedian_95++;
                }
            }
        }
    }
    if (obs->GetN()) { obs->Sort(); bands->WriteTObject(obs); std::cout << " imported " << obs->GetName() << " with " << obs->GetN() << " points." << std::endl; }
    if (median->GetN()) { median->Sort(); bands->WriteTObject(median); std::cout << " imported " << median->GetName() << " with " << median->GetN() << " points." << std::endl; }
    if (median_95->GetN()) { median_95->Sort(); bands->WriteTObject(median_95); std::cout << " imported " << median_95->GetName() << " with " << median_95->GetN() << " points." << std::endl; }
}
void importLandS(TDirectory *bands, TString name, TString fileName, bool doObserved=true, bool doExpected=true) {
    TFile *in = TFile::Open(fileName);
    if (in == 0) { std::cerr << "Cannot open " << fileName << std::endl; return;  }
    importLandS(bands, name, in, doObserved, doExpected);
    in->Close();
}

double smoothWithPolyFit(double x, int npar, int n, double *xi, double *yi) {
    TVectorD fitRes = polyFit(x, yi[n/2], npar, n, xi, yi);
    return fitRes(0)+yi[n/2];
}

void printValueFromScan1D(TDirectory *bands, TString name, TString out) {
    TGraph *graph = (TGraph*) bands->Get(name);
    if (graph == 0) return;
    double *x = graph->GetX();
    double *y = graph->GetY();
    int imin = 0, n = graph->GetN();
    for (int i = 1; i < n; ++i) {
        if (y[i] < y[imin]) imin = i;
    }
    double t1 = 1, t2 = 3.84;
    bool   hi68ok = false, hi95ok = false, lo68ok = false, lo95ok = false;
    double hi68 = x[n-1], hi95 = x[n-1], lo68 = x[0], lo95 = x[0];
    for (int i = 0; i < n-1; ++i) {
        if (y[i] > t1 && y[i+1] < t1) {
            double d1 = fabs(y[i] - t1), d2 = fabs(y[i+1] - t1);
            lo68 = (x[i]*d2 + x[i+1]*d1)/(d1+d2); lo68ok = true; 
        } else if (y[i] < t1 && y[i+1] > t1) {
            double d1 = fabs(y[i] - t1), d2 = fabs(y[i+1] - t1);
            hi68 = (x[i]*d2 + x[i+1]*d1)/(d1+d2); hi68ok = true; 
        }
        if (y[i] > t2 && y[i+1] < t2) {
            double d1 = fabs(y[i] - t2), d2 = fabs(y[i+1] - t2);
            lo95 = (x[i]*d2 + x[i+1]*d1)/(d1+d2); lo95ok = true; 
        } else if (y[i] < t2 && y[i+1] > t2) {
            double d1 = fabs(y[i] - t2), d2 = fabs(y[i+1] - t2);
            hi95 = (x[i]*d2 + x[i+1]*d1)/(d1+d2); hi95ok = true; 
        }
    }
    FILE *log = fopen(out.Data(), "w");
    fprintf(log, "Lowest point :  % 8.4f \n", x[imin]);
    if (lo68ok) fprintf(log, "Crossing at 1.00 from left:  % 8.4f \n", lo68);
    if (hi68ok) fprintf(log, "Crossing at 1.00 from right: % 8.4f \n", hi68);
    if (lo95ok) fprintf(log, "Crossing at 3.84 from left:  % 8.4f \n", lo95);
    if (hi95ok) fprintf(log, "Crossing at 3.84 from right: % 8.4f \n", hi95);
    fclose(log);
}

void array_sort(double *begin, double *end) { std::sort(begin, end); }
void array_sort(float *begin, float *end) { std::sort(begin, end); }
void array_sort(int *begin, int *end) { std::sort(begin, end); }
void array_sort(double &begin, double &end) { std::sort(&begin, &end); }
void array_sort(float &begin, float &end) { std::sort(&begin, &end); }
void array_sort(int &begin, int &end) { std::sort(&begin, &end); }

void bandUtils() {
}
