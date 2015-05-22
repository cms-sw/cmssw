TString base = "1";
TString scut = "1";
TString bcut = "1";
TTree   *stree = 0, *btree = 0;
Long64_t nMax = 999999999;
const int nrocs = 5, nwps = 10;
TGraph *rocs[nrocs]; TString lrocs[nrocs];
TGraph *wps[nwps]; TString lwps[nwps];
TH1    *frame = new TH1F("frame","frame;eff(background);eff(signal)",1000,0,1);
TLegend *legend = new TLegend(.5,.5,.8,.2);

void reset() {
    for (int i = 0; i < nrocs; ++i) rocs[i] = 0;
    for (int i = 0; i < nwps; ++i) wps[i] = 0;
}
void redraw() {
    legend->Clear();
    frame->Draw();
    for (int i = 0; i < nrocs; ++i) if (rocs[i]) {
        rocs[i]->Draw("L");
        legend->AddEntry(rocs[i],rocs[i]->GetTitle(),"L");
    }
    for (int i = 0; i < nwps; ++i) if (wps[i]) {
        wps[i]->Draw("P");
        legend->AddEntry(wps[i],wps[i]->GetTitle(),"P");
    }
    legend->Draw(); 
}

void opt1d_bins_fixEff(float effTarget, TString expr, bool highIsGood) {
    TString signal     = base+"&&"+scut;
    TString background = base+"&&"+bcut;
    double nS = stree->Draw(expr, signal, "", nMax);
    TH2F *hS = (TH2*) gROOT->FindObject("htemp")->Clone("hS");
    if (expr.Contains(">>htemp")) btree->Draw(expr, background, "", nMax);
    else btree->Draw(expr+TString::Format(">>htemp(%d,%g,%g)", hS->GetNbinsX(), hS->GetXaxis()->GetXmin(), hS->GetXaxis()->GetXmax(),hS->GetNbinsY(), hS->GetYaxis()->GetXmin(), hS->GetYaxis()->GetXmax()), background, "", nMax);
    TH2F *hB = (TH1*) gROOT->FindObject("htemp")->Clone("hB");
    float totS = hS->GetEntries(), totB = hB->GetEntries();
    for (unsigned int ix = 1, nx = hS->GetNbinsX(); ix <= nx; ++ix) {
        for (unsigned int iy = 1, ny = hS->GetNbinsY(); iy <= ny+1; ++iy) {
            unsigned int ybin = (!highIsGood ? iy : ny-iy+1), yprev = (!highIsGood ? ybin-1 : ybin+1);
            hS->SetBinContent(ix, ybin, hS->GetBinContent(ix,ybin) + hS->GetBinContent(ix,yprev));
            hB->SetBinContent(ix, ybin, hB->GetBinContent(ix,ybin) + hB->GetBinContent(ix,yprev));
        }
        for (unsigned int iy = 1, ny = hS->GetNbinsY(); iy <= ny; ++iy) {
            unsigned int ybin = (!highIsGood ? iy : ny-iy+1), ylast = (!highIsGood ? ny+1 : 0);
            hS->SetBinContent(ix, ybin, hS->GetBinContent(ix,ybin)/hS->GetBinContent(ix,ylast));
            hB->SetBinContent(ix, ybin, hB->GetBinContent(ix,ybin)/hB->GetBinContent(ix,ylast));
            //printf("for X bin %2d, Y bin %2d: sig eff = %.3f, bkg eff = %.3f, stot = %.0f, btot = %.0f\n", ix, ybin, hS->GetBinContent(ix,ybin),hB->GetBinContent(ix,ybin),hS->GetBinContent(ix,ylast),hB->GetBinContent(ix,ylast));
        }
    }
    float selS = 0, selB = 0;
    for (unsigned int ix = 1, nx = hS->GetNbinsX(); ix <= nx; ++ix) {
        for (unsigned int iy = 1, ny = hS->GetNbinsY(); iy <= ny; ++iy) {
            unsigned int ybin = (highIsGood ? iy : ny-iy+1), yprev = (highIsGood ? ybin-1 : ybin+1), ylast = (!highIsGood ? ny+1 : 0);
            //printf("for X bin %2d: Y bin %2d: sig eff = %.3f (prev bin %d eff = %.3f)\n", ix, ybin, hS->GetBinContent(ix,ybin), yprev, yprev != ylast ? hS->GetBinContent(ix,yprev) : 1.0);
            if (hS->GetBinContent(ix,ybin) < effTarget && (yprev == ylast || hS->GetBinContent(ix,yprev) >= effTarget)) {
                float ycut = !highIsGood ? hS->GetYaxis()->GetBinUpEdge(yprev) : hS->GetYaxis()->GetBinLowEdge(yprev);
                float effS = hS->GetBinContent(ix,yprev), evS = hS->GetBinContent(ix,ylast);
                float effB = hB->GetBinContent(ix,yprev), evB = hB->GetBinContent(ix,ylast);
                printf("for X bin %2d: Y bin %2d: cut Y at %8.3f: sig eff = %.3f, bkg eff = %.3f, log(S/B) = %.4f (before cut: %.4f)\n", ix, yprev, ycut, effS, effB, log((effS*evS)/(effB*evB)), log(evS/evB));
                selS += effS*evS; selB += effB*evB;
                break;
            }
        }
    }
    printf("overall: sig eff = %.3f, bkg eff = %.3f, S/B = %.4f (before cut: %.4f)\n", selS/totS, selB/totB, log(selS/selB), log(totS/totB));
}
void opt1d_2bins_BF(float effTarget, TString expr, bool highIsGood) {
    TString signal     = base+"&&"+scut;
    TString background = base+"&&"+bcut;
    double nS = stree->Draw(expr, signal, "", nMax);
    TH2F *hS = (TH2*) gROOT->FindObject("htemp")->Clone("hS");
    if (hs->GetNbinsX() != 2) { std::cerr << "This macro assumes only two X bins" << std::endl; return; }
    if (expr.Contains(">>htemp")) btree->Draw(expr, background, "", nMax);
    else btree->Draw(expr+TString::Format(">>htemp(%d,%g,%g)", hS->GetNbinsX(), hS->GetXaxis()->GetXmin(), hS->GetXaxis()->GetXmax(),hS->GetNbinsY(), hS->GetYaxis()->GetXmin(), hS->GetYaxis()->GetXmax()), background, "", nMax);
    TH2F *hB = (TH1*) gROOT->FindObject("htemp")->Clone("hB");
    float totS = hS->GetEntries(), totB = hB->GetEntries();
    unsigned int  nx = hS->GetNbinsX(), ny = hS->GetNbinsY(), ylast = (!highIsGood ? ny+1 : 0), yfirst = (!highIsGood ? 0 : ny+1);
    for (unsigned int ix = 1; ix <= nx; ++ix) {
        for (unsigned int iy = 1; iy <= ny+1; ++iy) {
            unsigned int ybin = (!highIsGood ? iy : ny-iy+1), yprev = (!highIsGood ? ybin-1 : ybin+1);
            hS->SetBinContent(ix, ybin, hS->GetBinContent(ix,ybin) + hS->GetBinContent(ix,yprev));
            hB->SetBinContent(ix, ybin, hB->GetBinContent(ix,ybin) + hB->GetBinContent(ix,yprev));
        }
        }
    float selS = 0, selB = 0;
    float targetS = effTarget * totS;
    float bestSOB = 0; unsigned int best_ybin[4];
    for (unsigned int iy = 1; iy <= ny; ++iy) {
        unsigned int ybin1 = (!highIsGood ? iy : ny-iy+1);
        float selS1 = hS->GetBinContent(1, ybin1);
        //printf("for X bin %2d: Y bin %2d: sig eff = %.3f\n", 1, ybin1, selS1/totS);
        for (unsigned int iy2 = 1; iy2 <= ny; ++iy2) {
            unsigned int ybin2 = (!highIsGood ? iy2 : ny-iy2+1);
            float selS2 = hS->GetBinContent(2, ybin2);
            selS = selS1 + selS2;
            //printf("   for X bin %2d: Y bin %2d: sig eff = %.3f, total = %.3f\n", 2, ybin2, selS2/totS, selS/totS);
            if (selS < targetS) continue;
            float selB = hB->GetBinContent(1, ybin1) +  hB->GetBinContent(2, ybin2);
            //printf("        bkg eff = %.3f, log(S/B) = %.3f\n", selB, log((selS/selB)));
            if (selS/selB > bestSOB) {
                bestSOB = selS/selB; best_ybin[1] = ybin1; best_ybin[2] = ybin2;
            }
            break;
        }
    }

    selS = 0; selB = 0;
    for (unsigned int ix = 1; ix <= nx; ++ix) {
        unsigned int yprev = (!highIsGood ? best_ybin[ix]-1 : best_ybin[ix]+1);
        float ycut = !highIsGood ? hS->GetYaxis()->GetBinUpEdge(yprev) : hS->GetYaxis()->GetBinLowEdge(yprev);
        float myS = hS->GetBinContent(ix,yprev); float mytotS = hS->GetBinContent(ix,ylast);
        float myB = hB->GetBinContent(ix,yprev); float mytotB = hB->GetBinContent(ix,ylast);
        selS += myS; selB += myB;
        printf("for X bin %2d: Y bin %2d: cut Y at %8.3f: sig eff = %.3f, bkg eff = %.3f, log(S/B) = %.4f (before cut: %.4f)\n", ix, best_ybin[ix], ycut, myS/mytotS, myB/mytotB, log(myS/myB), log(mytotS/mytotB));
    }
    printf("overall: sig eff = %.3f, bkg eff = %.3f, S/B = %.4f (before cut: %.4f)\n", selS/totS, selB/totB, log(selS/selB), log(totS/totB));
}

void opt2d_fixEff(float effTarget, TString expr, bool highXIsGood, bool highYIsGood, bool doAnd=true) {
    if (!doAnd) { std::cerr << "Sorry, only AND supported for now." << std::endl; return; }
    TString signal     = base+"&&"+scut;
    TString background = base+"&&"+bcut;
    double nS = stree->Draw(expr, signal, "", nMax);
    TH2F *hS = (TH2*) gROOT->FindObject("htemp")->Clone("hS");
    if (expr.Contains(">>htemp")) btree->Draw(expr, background, "", nMax);
    else btree->Draw(expr+TString::Format(">>htemp(%d,%g,%g)", hS->GetNbinsX(), hS->GetXaxis()->GetXmin(), hS->GetXaxis()->GetXmax(),hS->GetNbinsY(), hS->GetYaxis()->GetXmin(), hS->GetYaxis()->GetXmax()), background, "", nMax);
    TH2F *hB = (TH1*) gROOT->FindObject("htemp")->Clone("hB");
    double totS = hS->GetEntries(), totB = hB->GetEntries();
    double starget = totS * effTarget;
    unsigned int x0 = highXIsGood ? hS->GetNbinsX() : 1, dx = highXIsGood ? -1 : +1, xend = highXIsGood ? 0 : hS->GetNbinsX()+1;
    unsigned int y0 = highYIsGood ? hS->GetNbinsY() : 1, dy = highYIsGood ? -1 : +1, yend = highYIsGood ? 0 : hS->GetNbinsY()+1;
    //printf("\n---- PASS 0 ----\n");
    //for (unsigned int ix = x0-dx; ix != xend; ix += dx) {
    //    for (unsigned int iy = y0; iy != yend; iy += dy) {
    //        printf("for X bin %2d, Y bin %2d: sig = %.3f, bkg = %.3f\n", ix, iy, hS->GetBinContent(ix,iy),hB->GetBinContent(ix,iy));
    //    }
    //}
    //printf("\n---- PASS 1 ----\n");
    for (unsigned int ix = x0-dx; ix != xend; ix += dx) {
        for (unsigned int iy = y0; iy != yend; iy += dy) {
            hS->SetBinContent(ix, iy, hS->GetBinContent(ix,iy) + hS->GetBinContent(ix,iy-dy));
            hB->SetBinContent(ix, iy, hB->GetBinContent(ix,iy) + hB->GetBinContent(ix,iy-dy));
            //printf("for X bin %2d, Y bin %2d: sig = %.3f, bkg = %.3f\n", ix, iy, hS->GetBinContent(ix,iy),hB->GetBinContent(ix,iy));
        }
    }
    //printf("\n---- PASS 2 ----\n");
    for (unsigned int iy = y0-dy; iy != yend; iy += dy) {
        for (unsigned int ix = x0; ix != xend; ix += dx) {
            hS->SetBinContent(ix, iy, hS->GetBinContent(ix,iy) + hS->GetBinContent(ix-dx,iy));
            hB->SetBinContent(ix, iy, hB->GetBinContent(ix,iy) + hB->GetBinContent(ix-dx,iy));
            //printf("for X bin %2d, Y bin %2d: sig = %.3f, bkg = %.3f\n", ix, iy, hS->GetBinContent(ix,iy),hB->GetBinContent(ix,iy));
        }
    }
    //printf("\n---- PASS 3 ----\n");
    unsigned int bestx = x0, besty = y0; double bestb = totB, bests = 0;
    for (unsigned int ix = dx; ix != xend; ix += dx) {
        for (unsigned int iy = y0; iy != yend; iy += dy) {
            if (hS->GetBinContent(ix,iy) >= starget) {
                if (hB->GetBinContent(ix,iy) < bestb) {
                    bestx = ix; besty = iy;
                    bestb = hB->GetBinContent(ix,iy);
                    bests = hS->GetBinContent(ix,iy);
                    //printf("for X bin %2d, Y bin %2d: sig = %.3f, bkg = %.3f\n", ix, iy, hS->GetBinContent(ix,iy),hB->GetBinContent(ix,iy));
                }
            }
        }
    }
    printf("Optimal point: x %s %.4f %s y %s %.4f: sig eff = %.3f, bkg eff = %.3f, S/B = %.4f (before cut: %.4f)\n", 
        highXIsGood ? ">=" : "<=", highXIsGood ? hS->GetXaxis()->GetBinLowEdge(bestx) : hS->GetXaxis()->GetBinUpEdge(bestx),    
        doAnd ? "&&" : "||",
        highYIsGood ? ">=" : "<=", highYIsGood ? hS->GetYaxis()->GetBinLowEdge(besty) : hS->GetYaxis()->GetBinUpEdge(besty),    
        bests/totS, bestb/totB, log(bests/bestb), log(totS/totB));
}

float _getEffS(TString expr) {
    double nS = stree->Draw(expr+">>htemp(2,-0.5,1.5)", base+"&&"+scut, "GOFF", nMax);
    TH1* htemp = (TH1*) gROOT->FindObject("htemp");
    return htemp->GetBinContent(2)/(htemp->GetBinContent(1)+htemp->GetBinContent(2));
}
float _getEffB(TString expr) {
    double nS = btree->Draw(expr+">>htemp(2,-0.5,1.5)", base+"&&"+bcut, "GOFF", nMax);
    TH1* htemp = (TH1*) gROOT->FindObject("htemp");
    return htemp->GetBinContent(2)/(htemp->GetBinContent(1)+htemp->GetBinContent(2));
}
TString _makeExpr(TString templ, int nparams, float *vals) {
    char buff0[4],buff1[100];
    TString ret = templ.Data();
    for (int i = 0; i < nparams; ++i) {
        sprintf(buff0,"{%d}",i);
        sprintf(buff1,vals[i] < 0 ? "(%g)" : "%g",vals[i]);
        ret.ReplaceAll(buff0,buff1);
    }
    return ret;
} 

void optND_bf_fixEff(float effTarget, TString expr, int nparams, TString params, TString algo="Random", int steps=20,int verbose=1) {
    float xval[10],xmin[10],xmax[10],best[10],bestB=1.0;
    TObjArray *splits = params.Tokenize(";");
    if (nparams > 10) { std::cout << "Error: more than 10 params." << std::endl; return; }
    if (splits->GetEntries() != nparams) { std::cout << "Error: found " << splits->GetEntries() << " tokens for " << nparams << " params." << std::endl; return; }
    for (int i = 0; i < nparams; ++i) {
        TString sparam = ((TObjString*) splits->At(i))->GetString();
        std::string str = sparam.Data();
        int i1 = str.find(',');
        if (i1 == int(std::string::npos)) { std::cerr << "Error: bad token " << str << std::endl; return; }
        int i2 = str.find(',',i1+1);
        if (i2 == int(std::string::npos)) { std::cerr << "Error: bad token " << str << std::endl; return; }
        xval[i] = atof(str.substr(0,i1).c_str());
        xmin[i] = atof(str.substr(i1+1,i2-i1-1).c_str());
        xmax[i] = atof(str.substr(i2+1).c_str());
        printf("param %d: %g [ %g , %g ]\n",i,xval[i],xmin[i],xmax[i]);
    }
    delete splits;
    // Get starting point
    if (algo == "Random") {
        for (int iter = 0; iter < steps; ++iter) {
            if (verbose>1) printf("iteration %d\n",iter);
            for (int i = 0; i < nparams; ++i) {
                xval[i] = xmin[i] + gRandom->Rndm()*(xmax[i]-xmin[i]);
                if (verbose>1) printf("\tparam %d: %g [ %g , %g ]\n",i,xval[i],xmin[i],xmax[i]);
            }
            TString myexpr = _makeExpr(expr,nparams,xval);
            float effS = _getEffS(myexpr);
            if (verbose>1) printf("\teff S: %.4f\n", effS);
            if (effS < effTarget) continue;
            float effB = _getEffB(myexpr);
            if (verbose>1) printf("\teff B: %.4f\n", effB);
            if (effB < bestB)  {
                bestB = effB;
                for (int i = 0; i < nparams; ++i) best[i] = xval[i];
                if (verbose == 1) {
                    printf("\titer %4d: eff. sig = %.4f (target: %.4f), eff. bkg = %.4f, cut: %s\n", iter,effS,effTarget,effB,myexpr.Data());
                }
            }
        };
    } else if (algo == "ScanND") {
        int xpoints = ceil(pow(steps,1.0/nparams));
        int npoints = pow(xpoints,nparams);
        for (int iter = 0; iter < npoints; ++iter) {
            if (verbose>1) printf("iteration %d\n",iter);
            int icoord = iter;
            for (int i = 0; i < nparams; ++i) {
                int imy = icoord % xpoints; icoord /= nparams;
                xval[i] = xmin[i] + (imy+0.5)/xpoints*(xmax[i]-xmin[i]);
                if (verbose>1) printf("\tparam %d: %g [ %g , %g ]\n",i,xval[i],xmin[i],xmax[i]);
            }
            TString myexpr = _makeExpr(expr,nparams,xval);
            float effS = _getEffS(myexpr);
            if (verbose>1) printf("\teff S: %.4f\n", effS);
            if (effS < effTarget) continue;
            float effB = _getEffB(myexpr);
            if (verbose>1) printf("\teff B: %.4f\n", effB);
            if (effB < bestB)  {
                bestB = effB;
                for (int i = 0; i < nparams; ++i) best[i] = xval[i];
                if (verbose == 1) {
                    printf("\titer %4d: eff. sig = %.4f (target: %.4f), eff. bkg = %.4f, cut: %s\n", iter,effS,effTarget,effB,myexpr.Data());
                }
            }
        }
    }
    std::cout << "Best cut: " << _makeExpr(expr,nparams,best) << " with background efficiency " << bestB << std::endl;
}


void cutstudy() {
    reset();
    frame->GetYaxis()->SetRangeUser(0,1);
    legend->SetTextFont(42);
    legend->SetTextSize(0.04);
    legend->SetFillColor(0);
}

void newRoc(int iroc, TString expr, bool highIsGood, TString label, int color, int width=2, int style=1) {
    rocs[iroc] = rocCurve2(expr,stree,base+" && "+scut,btree,base+" && "+bcut, highIsGood, nMax);
    rocs[iroc]->SetTitle(label);
    rocs[iroc]->SetLineColor(color);
    rocs[iroc]->SetLineWidth(width);
    rocs[iroc]->SetLineStyle(style);
    redraw();
}
void newWP(int iwp, TString expr, TString label, int color, float size=1.2, int style=20) {
    wps[iwp] = rocCurve2(expr,stree,base+" && "+scut,btree,base+" && "+bcut, 1, nMax);
    wps[iwp]->SetTitle(label);
    wps[iwp]->SetMarkerColor(color);
    wps[iwp]->SetMarkerSize(size);
    wps[iwp]->SetMarkerStyle(style);
    redraw();
}
