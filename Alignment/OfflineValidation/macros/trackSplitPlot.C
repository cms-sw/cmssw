/***********************************
Table Of Contents
0. Track Split Plot
1. Misalignment Dependence
2. Make Plots
3. Axis Label
4. Axis Limits
5. Place Legend
6. TDR Style
***********************************/

#include <vector>
#include "trackSplitPlot.h"

//===================
//0. Track Split Plot
//===================

TCanvas *trackSplitPlot(Int_t nFiles,TString *files,TString *names,TString xvar,TString yvar,
                        Bool_t relative,Bool_t resolution,Bool_t pull,
                        TString saveas)
{
    stufftodelete->SetOwner(true);
    cout << xvar << " " << yvar << endl;
    if (xvar == "" && yvar == "")
        return 0;

    PlotType type;
    if (xvar == "")      type = Histogram;
    else if (yvar == "") type = OrgHistogram;
    else if (resolution) type = Resolution;
    else if (nFiles < 1) type = ScatterPlot;
    else                 type = Profile;
    if (nFiles < 1) nFiles = 1;

    const Int_t n = nFiles;

    setTDRStyle();
    gStyle->SetOptStat(0);        //for histograms, the mean and rms are included in the legend
    //for a scatterplot, this is needed to show the z axis scale
    //for non-pull histograms or when run number is on the x axis, this is needed so that 10^-? on the right is not cut off
    if (type == ScatterPlot || (type == Histogram && !pull) || xvar == "runNumber")
    {
        gStyle->SetCanvasDefW(678);
        gStyle->SetPadRightMargin(0.115);
    }
    else
    {
        gStyle->SetCanvasDefW(600);
        gStyle->SetPadRightMargin(0.04);
    }

    Bool_t nHits = (xvar[0] == 'n' && xvar[1] == 'H' && xvar[2] == 'i'    //This includes nHits, nHitsTIB, etc.
                                   && xvar[3] == 't' && xvar[4] == 's');

    Int_t nBinsScatterPlotx = binsScatterPlotx;
    Int_t nBinsScatterPloty = binsScatterPloty;
    Int_t nBinsHistogram = binsHistogram;
    Int_t nBinsProfileResolution = binsProfileResolution;
    if (xvar == "runNumber")
    {
        nBinsProfileResolution = runNumberBins;
        nBinsHistogram = runNumberBins;
    }
    if (nHits)
    {
        nBinsHistogram = (int)(findMax(nFiles,files,xvar,'x') - findMin(nFiles,files,xvar,'x') + 1.1);     //in case it's .99999
        nBinsScatterPlotx = nBinsHistogram;
        nBinsProfileResolution = nBinsHistogram;
    }

    vector<TH1*> p;
    Int_t lengths[n];

    stringstream sx,sy,srel,ssigma1,ssigma2,ssigmaorg;

    sx << xvar << "_org";
    TString xvariable = sx.str();
    TString xvariable2 = "";
    if (xvar == "runNumber") xvariable = "runNumber";
    if (nHits)
    {
        xvariable  = xvar;
        xvariable2 = xvar;
        xvariable.Append("1_spl");
        xvariable2.Append("2_spl");
    }

    sy << "Delta_" << yvar;
    TString yvariable = sy.str();

    TString relvariable = "1";
    if (relative)
    {
        srel << yvar << "_org";
        relvariable = srel.str();
    }

    TString sigma1variable = "",sigma2variable = "";
    if (pull)
    {
        ssigma1 << yvar << "1Err_spl";
        ssigma2 << yvar << "2Err_spl";
    }
    sigma1variable = ssigma1.str();
    sigma2variable = ssigma2.str();

    TString sigmaorgvariable = "";
    if (pull && relative)
        ssigmaorg << yvar << "Err_org";
    sigmaorgvariable = ssigmaorg.str();


    Double_t xmin = -1,xmax = 1,ymin = -1,ymax = 1;
    if (type == Profile || type == ScatterPlot || type == OrgHistogram || type == Resolution)
        axislimits(nFiles,files,xvar,'x',relative,pull,xmin,xmax);
    if (type == Profile || type == ScatterPlot || type == Histogram || type == Resolution)
        axislimits(nFiles,files,yvar,'y',relative,pull,ymin,ymax);

    std::vector<TString> meansrmss(n);
    Bool_t  used[n];        //a file is not "used" if it's MC data and the x variable is run number, or if the filename is blank

    for (Int_t i = 0; i < n; i++)
    {
        stringstream sid;
        sid << "p" << i;
        TString id = sid.str();

        //for a profile or resolution, it fills a histogram, q[j], for each bin, then gets the mean and width from there.
        vector<TH1F*> q;

        if (type == ScatterPlot)
            p.push_back(new TH2F(id,"",nBinsScatterPlotx,xmin,xmax,nBinsScatterPloty,ymin,ymax));
        if (type == Histogram)
            p.push_back(new TH1F(id,"",nBinsHistogram,ymin,ymax));
        if (type == OrgHistogram)
            p.push_back(new TH1F(id,"",nBinsHistogram,xmin,xmax));
        if (type == Resolution || type == Profile)
        {
            p.push_back(new TH1F(id,"",nBinsProfileResolution,xmin,xmax));
            for (Int_t j = 0; j < nBinsProfileResolution; j++)
            {

                stringstream sid2;
                sid2 << "q" << i << j;
                TString id2 = sid2.str();
                q.push_back(new TH1F(id2,"",nBinsHistogram,ymin,ymax));

            }
        }
        stufftodelete->Add(p[i]);
        p[i]->SetBit(kCanDelete,true);

        used[i] = true;
        if ((xvar == "runNumber" ? findMax(files[i],"runNumber",'x') < 2 : false) || files[i] == "")  //if it's MC data (run 1), the run number is meaningless
        {
            used[i] = false;
            p[i]->SetLineColor(kWhite);
            p[i]->SetMarkerColor(kWhite);
            for (unsigned int j = 0; j < q.size(); j++)
                delete q[j];
            continue;
        }

        TFile *f = TFile::Open(files[i]);
        TTree *tree = (TTree*)f->Get("cosmicValidation/splitterTree");
        if (tree == 0)
            tree = (TTree*)f->Get("splitterTree");

        lengths[i] = tree->GetEntries();

        Double_t x = 0, y = 0, rel = 1, sigma1 = 1, sigma2 = 1,           //if !pull, we want to divide by sqrt(2) because we want the error from 1 track
                                                  sigmaorg = 0;
        Int_t xint = 0, xint2 = 0;
        Int_t runNumber = 0;

        if (!relative && !pull && (yvar == "dz" || yvar == "dxy"))
            rel = 1e-4;                                     //it's in cm but we want it in um, so divide by 1e-4

        tree->SetBranchAddress("runNumber",&runNumber);
        if (type == Profile || type == ScatterPlot || type == Resolution || type == OrgHistogram)
        {
            if (xvar == "runNumber")
                tree->SetBranchAddress(xvariable,&xint);
            else if (nHits)
            {
                tree->SetBranchAddress(xvariable,&xint);
                tree->SetBranchAddress(xvariable2,&xint2);
            }
            else
                tree->SetBranchAddress(xvariable,&x);
        }
        if (type == Profile || type == ScatterPlot || type == Resolution || type == Histogram)
        {
            int branchexists = tree->SetBranchAddress(yvariable,&y);
            if (branchexists == -5)   //i.e. it doesn't exist
            {
                yvariable.ReplaceAll("Delta_","d");
                yvariable.Append("_spl");
                tree->SetBranchAddress(yvariable,&y);
            }
        }
        if (relative && xvar != yvar)                       //if xvar == yvar, setting the branch here will undo setting it to x 2 lines earlier
            tree->SetBranchAddress(relvariable,&rel);       //setting the value of rel is then taken care of later: rel = x
        if (pull)
        {
            tree->SetBranchAddress(sigma1variable,&sigma1);
            tree->SetBranchAddress(sigma2variable,&sigma2);
        }
        if (relative && pull)
            tree->SetBranchAddress(sigmaorgvariable,&sigmaorg);

        Int_t notincluded = 0;                              //this counts the number that aren't in the right run range.
                                                            //it's subtracted from lengths[i] in order to normalize the histograms

        for (Int_t j = 0; j<lengths[i]; j++)
        {
            tree->GetEntry(j);
            if (xvar == "runNumber" || nHits)
                x = xint;
            if (xvar == "runNumber")
                runNumber = x;
            if (yvar == "phi" && y >= pi)
                y -= 2*pi;
            if (yvar == "phi" && y <= -pi)
                y += 2*pi;
            if ((runNumber < minrun && runNumber > 1) || (runNumber > maxrun && maxrun > 0))  //minrun and maxrun are global variables.
            {
                notincluded++;
                continue;
            }
            if (relative && xvar == yvar)
                rel = x;
            Double_t error = 0;
            if (relative && pull)
                error = sqrt((sigma1/rel)*(sigma1/rel) + (sigma2/rel)*(sigma2/rel) + (sigmaorg*y/(rel*rel))*(sigmaorg*x/(rel*rel)));
            else
                error = sqrt(sigma1 * sigma1 + sigma2 * sigma2);   // = sqrt(2) if !pull; this divides by sqrt(2) to get the error in 1 track
            y /= (rel * error);

            if (ymin <= y && y < ymax && xmin <= x && x < xmax)
            {
                if (type == Histogram)
                    p[i]->Fill(y);
                if (type == ScatterPlot)
                    p[i]->Fill(x,y);
                if (type == Resolution || type == Profile)
                {
                    int which = (p[i]->Fill(x,0)) - 1;
                    //get which q[j] by filling p[i] with nothing.  (TH1F::Fill returns the bin number)
                    //p[i]'s actual contents are set later.
                    if (which >= 0 && (unsigned)which < q.size()) q[which]->Fill(y);
                }
                if (type == OrgHistogram)
                    p[i]->Fill(x);
            }

            if (nHits)
            {
                x = xint2;
                if (ymin <= y && y < ymax && xmin <= x && x < xmax)
                {
                    if (type == Histogram)
                        p[i]->Fill(y);
                    if (type == ScatterPlot)
                        p[i]->Fill(x,y);
                    if (type == Resolution || type == Profile)
                    {
                        int which = (p[i]->Fill(x,0)) - 1;
                        if (which >= 0) q[which]->Fill(y);         //get which q[j] by filling p[i] (with nothing), which returns the bin number
                    }
                    if (type == OrgHistogram)
                        p[i]->Fill(x);
                }
            }

            if (lengths[i] < 10 ? true :
                (((j+1)/(int)(pow(10,(int)(log10(lengths[i]))-1)))*(int)(pow(10,(int)(log10(lengths[i]))-1)) == j + 1 || j + 1 == lengths[i]))
            //print when j+1 is a multiple of 10^x, where 10^x has 1 less digit than lengths[i]
            // and when it's finished
            //For example, if lengths[i] = 123456, it will print this when j+1 = 10000, 20000, ..., 120000, 123456
            //So it will print between 10 and 100 times: 10 when lengths[i] = 10^x and 100 when lengths[i] = 10^x - 1
            {
                cout << j + 1 << "/" << lengths[i] << ": ";
                if (type == Profile || type == ScatterPlot || type == Resolution)
                    cout << x << ", " << y << endl;
                if (type == OrgHistogram)
                    cout << x << endl;
                if (type == Histogram)
                    cout << y << endl;
            }
        }
        lengths[i] -= notincluded;

        meansrmss[i] = "";
        if (type == Histogram || type == OrgHistogram)
        {
            cout << "Average = " << p[i]->GetMean() << endl;
            cout << "RMS     = " << p[i]->GetRMS()  << endl;
            stringstream meanrms;
            meanrms.precision(3);
            meanrms << "#mu=" << p[i]->GetMean() << ", #sigma=" << p[i]->GetRMS();
            meansrmss[i] = meanrms.str();
        }

        if (type == Resolution)
        {
            for (Int_t j = 0; j < nBinsProfileResolution; j++)
            {
                p[i]->SetBinContent(j+1,q[j]->GetRMS());
                p[i]->SetBinError  (j+1,q[j]->GetRMSError());
                delete q[j];
            }
        }

        if (type == Profile)
        {
            for (Int_t j = 0; j < nBinsProfileResolution; j++)
            {
                p[i]->SetBinContent(j+1,q[j]->GetMean());
                p[i]->SetBinError  (j+1,q[j]->GetMeanError());
                delete q[j];
            }
        }

        setAxisLabels(p[i],type,xvar,yvar,relative,pull);

        p[i]->SetLineColor(colors[i]);
        p[i]->SetLineStyle(styles[i]);
        if (type == Resolution || type == Profile)
        {
            p[i]->SetMarkerColor(colors[i]);
            p[i]->SetMarkerStyle(20+i);
        }
        else
        {
            p[i]->SetMarkerColor(kWhite);
            p[i]->SetMarkerStyle(1);
        }
    }

    TH1 *firstp = 0;
    for (int i = 0; i < n; i++)
    {
        if (used[i])
        {
            firstp = p[i];
            break;
        }
    }
    if (firstp == 0)
    {
        stufftodelete->Clear();
        return 0;
    }

    TCanvas *c1 = TCanvas::MakeDefCanvas();

    TH1 *maxp = firstp;
    if (type == ScatterPlot)
        firstp->Draw("COLZ");
    else if (type == Resolution || type == Profile)
    {
        vector<TGraphErrors*> g;
        TMultiGraph *list = new TMultiGraph();
        for (Int_t i = 0, ii = 0; i < n; i++, ii++)
        {
            if (!used[i])
            {
                ii--;
                continue;
            }
            g.push_back(new TGraphErrors(p[i]));
            for (Int_t j = 0; j < g[ii]->GetN(); j++)
            {
                if (g[ii]->GetY()[j] == 0 && g[ii]->GetEY()[j] == 0)
                {
                    g[ii]->RemovePoint(j);
                    j--;
                }
            }
            list->Add(g[ii]);
        }
        list->Draw("AP");
        Double_t yaxismax = list->GetYaxis()->GetXmax();
        Double_t yaxismin = list->GetYaxis()->GetXmin();
        delete list;       //automatically deletes g[i]
        if (yaxismin > 0)
        {
            yaxismax += yaxismin;
            yaxismin = 0;
        }
        firstp->GetYaxis()->SetRangeUser(yaxismin,yaxismax);
        if (xvar == "runNumber")
            firstp->GetXaxis()->SetNdivisions(505);
    }
    else if (type == Histogram || type == OrgHistogram)
    {
        Bool_t allthesame = true;
        for (Int_t i = 1; i < n && allthesame; i++)
        {
            if (lengths[i] != lengths[0])
                allthesame = false;
        }
        if (!allthesame && xvar != "runNumber")
            for (Int_t i = 0; i < n; i++)
            {
                p[i]->Scale(1.0/lengths[i]);     //This does NOT include events that are out of the run number range (minrun and maxrun).
                                                 //It DOES include events that are out of the histogram range.
            }
        maxp = (TH1F*)firstp->Clone("maxp");
        stufftodelete->Add(maxp);
        maxp->SetBit(kCanDelete,true);
        maxp->SetLineColor(kWhite);
        for (Int_t i = 1; i <= maxp->GetNbinsX(); i++)
        {
            for (Int_t j = 0; j < n; j++)
            {
                if (!used[j])
                    continue;
                maxp->SetBinContent(i,TMath::Max(maxp->GetBinContent(i),p[j]->GetBinContent(i)));
            }
        }
        maxp->SetMinimum(0);
        maxp->Draw();
        if (xvar == "runNumber")
        {
            maxp->GetXaxis()->SetNdivisions(505);
            maxp->Draw();
        }
    }

    TLegend *legend = new TLegend(.6,.7,.9,.9,"","br");
    stufftodelete->Add(legend);
    legend->SetBit(kCanDelete,true);
    if (n == 1 && !used[0])
    {
        deleteCanvas(c1);
        stufftodelete->Clear();
        return 0;
    }
    for (Int_t i = 0; i < n; i++)
    {
        if (!used[i])
            continue;
        if (type == Resolution || type == Profile)
        {
            if (p[i] == firstp)
                p[i]->Draw("P");
            else
                p[i]->Draw("same P");
            legend->AddEntry(p[i],names[i],"pl");
        }
        else if (type == Histogram || type == OrgHistogram)
        {
            p[i]->Draw("same");
            legend->AddEntry(p[i],names[i],"l");
            legend->AddEntry((TObject*)0,meansrmss[i],"");
        }
    }
    if (legend->GetListOfPrimitives()->At(0) == 0)
    {
        stufftodelete->Clear();
        deleteCanvas(c1);
        return 0;
    }


    c1->Update();
    Double_t x1min  = .98*gPad->GetUxmin() + .02*gPad->GetUxmax();
    Double_t x2max  = .02*gPad->GetUxmin() + .98*gPad->GetUxmax();
    Double_t y1min  = .98*gPad->GetUymin() + .02*gPad->GetUymax();
    Double_t y2max  = .02*gPad->GetUymin() + .98*gPad->GetUymax();
    Double_t width  = .4*(x2max-x1min);
    Double_t height = (1./20)*legend->GetListOfPrimitives()->GetEntries()*(y2max-y1min);
    if (type == Histogram || type == OrgHistogram)
    {
        width *= 2;
        height /= 2;
        legend->SetNColumns(2);
    }
    Double_t newy2max = placeLegend(legend,width,height,x1min,y1min,x2max,y2max);
    maxp->GetYaxis()->SetRangeUser(gPad->GetUymin(),(newy2max-.02*gPad->GetUymin())/.98);

    legend->SetFillStyle(0);
    legend->Draw();

    if (saveas != "")
        saveplot(c1,saveas);

    return c1;
}


//make a 1D histogram of Delta_yvar

TCanvas *trackSplitPlot(Int_t nFiles,TString *files,TString *names,TString var,
                        Bool_t relative,Bool_t pull,TString saveas)
{
    return trackSplitPlot(nFiles,files,names,"",var,relative,false,pull,saveas);
}



//For 1 file

TCanvas *trackSplitPlot(TString file,TString xvar,TString yvar,Bool_t profile,
                        Bool_t relative,Bool_t resolution,Bool_t pull,
                        TString saveas)
{
    Int_t nFiles = 0;
    if (profile)                       //it interprets nFiles < 1 as 1 file, make a scatterplot
        nFiles = 1;
    TString *files = &file;
    TString name = "";
    TString *names = &name;
    return trackSplitPlot(nFiles,files,names,xvar,yvar,relative,resolution,pull,saveas);
}

//make a 1D histogram of Delta_yvar

TCanvas *trackSplitPlot(TString file,TString var,
                        Bool_t relative,Bool_t pull,
                        TString saveas)
{
    Int_t nFiles = 1;
    TString *files = &file;
    TString name = "";
    TString *names = &name;
    return trackSplitPlot(nFiles,files,names,var,relative,pull,saveas);
}

void placeholder(TString saveas,Bool_t wide)
{
    setTDRStyle();
    if (wide)
        gStyle->SetCanvasDefW(678);
    else
        gStyle->SetCanvasDefW(600);
    TText line1(.5,.6,"This is a placeholder so that when there are");
    TText line2(.5,.4,"4 plots per line it lines up nicely");
    line1.SetTextAlign(22);
    line2.SetTextAlign(22);
    TCanvas *c1 = TCanvas::MakeDefCanvas();
    line1.Draw();
    line2.Draw();
    if (saveas != "")
        saveplot(c1,saveas);
    deleteCanvas(c1);
}

void saveplot(TCanvas *c1,TString saveas)
{
    if (saveas == "")
        return;
    TString saveas2 = saveas,
            saveas3 = saveas;
    saveas2.ReplaceAll(".pngepsroot","");
    saveas3.Remove(saveas3.Length()-11);
    if (saveas2 == saveas3)
    {
        c1->SaveAs(saveas.ReplaceAll(".pngepsroot",".png"));
        c1->SaveAs(saveas.ReplaceAll(".png",".eps"));
        c1->SaveAs(saveas.ReplaceAll(".eps",".root"));
        c1->SaveAs(saveas.ReplaceAll(".root",".pdf"));
    }
    else
    {
        c1->SaveAs(saveas);
    }
}

void deleteCanvas(TObject *canvas)
{
    if (canvas == 0) return;
    if (!canvas->InheritsFrom("TCanvas"))
    {
        delete canvas;
        return;
    }
    TCanvas *c1 = (TCanvas*)canvas;
    TList *list = c1->GetListOfPrimitives();
    list->SetOwner(true);
    list->Clear();
    delete c1;
}

//This makes a plot, of Delta_yvar vs. runNumber, zoomed in to between firstrun and lastrun.
//Each bin contains 1 run.
//Before interpreting the results, make sure to look at the histogram of run number (using yvar = "")
//There might be bins with very few events => big error bars,
//or just 1 event => no error bar

void runNumberZoomed(Int_t nFiles,TString *files,TString *names,TString yvar,
                     Bool_t relative,Bool_t resolution,Bool_t pull,
                     Int_t firstRun,Int_t lastRun,TString saveas)
{
    Int_t tempminrun = minrun;
    Int_t tempmaxrun = maxrun;
    Int_t tempbins = runNumberBins;
    minrun = firstRun;
    maxrun = lastRun;
    runNumberBins = (int)(findMax(nFiles,files,"runNumber",'x')
                        - findMin(nFiles,files,"runNumber",'x') + 1.001);
    trackSplitPlot(nFiles,files,names,"runNumber",yvar,relative,resolution,pull,saveas);
    minrun = tempminrun;
    maxrun = tempmaxrun;
    runNumberBins = tempbins;
}

//==========================
//1. Misalignment Dependence
//==========================

//This can do three different things:
// (1) if xvar == "", it will plot the mean (if !resolution) or width (if resolution) of Delta_yvar as a function
//     of the misalignment values, as given in values.  misalignment (e.g. sagitta, elliptical) will be used as the
//     x axis label.
// (2) if xvar != "", it will fit the profile/resolution to a function.  If parameter > 0, it will plot the parameter given by parameter as
//     a function of the misalignment.  parametername is used as the y axis label.  You can put a semicolon in parametername
//     to separate the name from the units.  Functionname describes the funciton, and is put in brackets in the y axis label.
//     For example, to fit to Delta_pt = [0]*(eta_org-[1]), you could use functionname = "#Deltap_{T} = A(#eta_{org}-B)",
//     parameter = 0, and parametername = "A;GeV".
// (3) if parameter < 0, it will draw the profile/resolution along with the fitted functions.
//     The parameter of interest is still indicated by parameter, which is transformed to -parameter - 1.
//     For example, -1 --> 0, -2 --> 1, -3 --> 2, ...
//     This parameter's value and error will be in the legend.  You still need to enter parametername and functionname,
//     because they will be used for labels.

//The best way to run misalignmentDependence is through makePlots.  If you want to run misalignmentDependence directly,
//the LAST function, all the way at the bottom of this file, is probably the most practical to use (for all three of these).


// The first function takes a canvas as its argument.  This canvas needs to have been produced with trackSplitPlot using
// the same values of xvar, yvar, relative, resolution, and pull or something strange could happen.

void misalignmentDependence(TCanvas *c1old,
                            Int_t nFiles,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                            TF1 *function,Int_t parameter,TString parametername,TString functionname,
                            Bool_t relative,Bool_t resolution,Bool_t pull,
                            TString saveas)
{
    if (c1old == 0) return;
    c1old = (TCanvas*)c1old->Clone("c1old");
    if (misalignment == "" || yvar == "") return;
    Bool_t drawfits = (parameter < 0);
    if (parameter < 0)
        parameter = -parameter - 1;   //-1 --> 0, -2 --> 1, -3 --> 2, ...
    TString yaxislabel = nPart(1,parametername);
    TString parameterunits = nPart(2,parametername);
    if (parameterunits != "")
        yaxislabel.Append(" (").Append(parameterunits).Append(")");
    TList *list = c1old->GetListOfPrimitives();
    //const int n = list->GetEntries() - 2 - (xvar == "");
    const int n = nFiles;

    setTDRStyle();
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
    gStyle->SetFitFormat("5.4g");
    gStyle->SetFuncColor(2);
    gStyle->SetFuncStyle(1);
    gStyle->SetFuncWidth(1);
    if (!drawfits)
    {
        gStyle->SetCanvasDefW(678);
        gStyle->SetPadRightMargin(0.115);
    }

    TH1 **p = new TH1*[n];
    TF1 **f = new TF1*[n];
    Bool_t used[n];
    for (Int_t i = 0; i < n; i++)
    {
        stringstream s0;
        s0 << "p" << i;
        TString pname = s0.str();
        p[i] = (TH1*)list->/*At(i+1+(xvar == ""))*/FindObject(pname);
        used[i] = (p[i] != 0);
        if (used[i])
            p[i]->SetDirectory(0);
        if (xvar == "")
            continue;
        stringstream s;
        s << function->GetName() << i;
        TString newname = s.str();
        f[i] = (TF1*)function->Clone(newname);
        stufftodelete->Add(f[i]);
    }

    Double_t *result = new Double_t[nFiles];
    Double_t *error  = new Double_t[nFiles];
    if (xvar == "")
    {
        yaxislabel = axislabel(yvar,'y',relative,resolution,pull);
        for (Int_t i = 0; i < nFiles; i++)
        {
            if (!used[i]) continue;
            if (!resolution)
            {
                result[i] = p[i]->GetMean();
                error[i]  = p[i]->GetMeanError();
            }
            else
            {
                result[i] = p[i]->GetRMS();
                error[i]  = p[i]->GetRMSError();
            }
            cout << result[i] << " +/- " << error[i] << endl;
        }
    }
    else
    {
        for (int i = 0; i < n; i++)
        {
            if (!used[i]) continue;
            f[i]->SetLineColor(colors[i]);
            f[i]->SetLineStyle(styles[i]);
            f[i]->SetLineWidth(1);
            p[i]->SetMarkerColor(colors[i]);
            p[i]->SetMarkerStyle(20+i);
            p[i]->SetLineColor(colors[i]);
            p[i]->SetLineStyle(styles[i]);
            p[i]->Fit(f[i],"IM");
            error[i]  = f[i]->GetParError (parameter);
            //the fits sometimes don't work if the parameters are constrained.
            //take care of the constraining here.
            //for sine, make the amplitude positive and the phase between 0 and 2pi.
            //unless the amplitude is the only parameter (eg sagitta theta theta)
            if (function->GetName() == TString("sine") && function->GetNumberFreeParameters() >= 2)
            {
                if (f[i]->GetParameter(0) < 0)
                {
                    f[i]->SetParameter(0,-f[i]->GetParameter(0));
                    f[i]->SetParameter(2,f[i]->GetParameter(2)+pi);
                }
                while(f[i]->GetParameter(2) >= 2*pi)
                    f[i]->SetParameter(2,f[i]->GetParameter(2)-2*pi);
                while(f[i]->GetParameter(2) < 0)
                    f[i]->SetParameter(2,f[i]->GetParameter(2)+2*pi);
            }
            result[i] = f[i]->GetParameter(parameter);
        }
    }


    TCanvas *c1 = TCanvas::MakeDefCanvas();

    if (drawfits && xvar != "" && yvar != "")
    {
        TString legendtitle = "[";
        legendtitle.Append(functionname);
        legendtitle.Append("]");
        TLegend *legend = new TLegend(.7,.7,.9,.9,legendtitle,"br");
        stufftodelete->Add(legend);
        TString drawoption = "";
        for (int i = 0; i < n; i++)
        {
            if (!used[i]) continue;
            p[i]->Draw(drawoption);
            f[i]->Draw("same");
            drawoption = "same";

            stringstream s;
            s.precision(3);
            s << nPart(1,parametername) << " = " <<  result[i] << " #pm " << error[i];
            if (parameterunits != "") s << " " << parameterunits;
            TString str = s.str();
            legend->AddEntry(p[i],names[i],"pl");
            legend->AddEntry(f[i],str,"l");
        }
        c1->Update();
        Double_t x1min  = .98*gPad->GetUxmin() + .02*gPad->GetUxmax();
        Double_t x2max  = .02*gPad->GetUxmin() + .98*gPad->GetUxmax();
        Double_t y1min  = .98*gPad->GetUymin() + .02*gPad->GetUymax();
        Double_t y2max  = .02*gPad->GetUymin() + .98*gPad->GetUymax();
        Double_t width  = .4*(x2max-x1min);
        Double_t height = (1./20)*legend->GetListOfPrimitives()->GetEntries()*(y2max-y1min);
        width *= 2;
        height /= 2;
        legend->SetNColumns(2);

        Double_t newy2max = placeLegend(legend,width,height,x1min,y1min,x2max,y2max);
        p[0]->GetYaxis()->SetRangeUser(gPad->GetUymin(),(newy2max-.02*gPad->GetUymin())/.98);

        legend->SetFillStyle(0);
        legend->Draw();
    }
    else
    {
        if (values == 0) return;

        Bool_t phasesmatter = false;
        if (misalignment == "elliptical" || misalignment == "sagitta" || misalignment == "skew")
        {
            if (phases == 0)
            {
                cout << "This misalignment has a phase, but you didn't supply the phases!" << endl
                     << "Can't produce plots depending on the misalignment value." << endl;
                return;
            }
            int firstnonzero = -1;
            for (Int_t i = 0; i < nFiles; i++)
            {
                if (values[i] == 0) continue;                    //if the amplitude is 0 the phase is arbitrary
                if (firstnonzero == -1) firstnonzero = i;
                if (phases[i] != phases[firstnonzero])
                    phasesmatter = true;
            }
        }

        if (!phasesmatter)
        {
            TGraphErrors *g = new TGraphErrors(nFiles,values,result,(Double_t*)0,error);
            g->SetName("");
            stufftodelete->Add(g);

            TString xaxislabel = "#epsilon_{";
            xaxislabel.Append(misalignment);
            xaxislabel.Append("}");
            g->GetXaxis()->SetTitle(xaxislabel);
            if (xvar != "")
            {
                yaxislabel.Append("   [");
                yaxislabel.Append(functionname);
                yaxislabel.Append("]");
            }
            g->GetYaxis()->SetTitle(yaxislabel);

            g->SetMarkerColor(colors[0]);
            g->SetMarkerStyle(20);

            g->Draw("AP");
            Double_t yaxismax = g->GetYaxis()->GetXmax();
            Double_t yaxismin = g->GetYaxis()->GetXmin();
            if (yaxismin > 0)
            {
                yaxismax += yaxismin;
                yaxismin = 0;
            }
            g->GetYaxis()->SetRangeUser(yaxismin,yaxismax);
            g->Draw("AP");
        }
        else
        {
            double *xvalues = new double[nFiles];
            double *yvalues = new double[nFiles];      //these are not physically x and y (except in the case of skew)
            for (int i = 0; i < nFiles; i++)
            {
                xvalues[i] = values[i] * cos(phases[i]);
                yvalues[i] = values[i] * sin(phases[i]);
            }
            TGraph2DErrors *g = new TGraph2DErrors(nFiles,xvalues,yvalues,result,(Double_t*)0,(Double_t*)0,error);
            g->SetName("");
            stufftodelete->Add(g);
            delete[] xvalues;        //A TGraph2DErrors has its own copy of xvalues and yvalues, so it's ok to delete these copies.
            delete[] yvalues;

            TString xaxislabel = "#epsilon_{";
            xaxislabel.Append(misalignment);
            xaxislabel.Append("}cos(#delta)");
            TString realyaxislabel = xaxislabel;
            realyaxislabel.ReplaceAll("cos(#delta)","sin(#delta)");
            g->GetXaxis()->SetTitle(xaxislabel);
            g->GetYaxis()->SetTitle(realyaxislabel);
            TString zaxislabel = /*"fake"*/yaxislabel;         //yaxislabel is defined earlier
            if (xvar != "")
            {
                zaxislabel.Append("   [");
                zaxislabel.Append(functionname);
                zaxislabel.Append("]");
            }
            g->GetZaxis()->SetTitle(zaxislabel);
            g->SetMarkerStyle(20);
            g->Draw("pcolerr");
        }
    }

    if (saveas != "")
    {
        saveplot(c1,saveas);
        delete[] p;
        delete[] f;
        delete[] result;
        delete[] error;
        delete c1old;
    }
}


//This version allows you to show multiple parameters.  It runs the previous version multiple times, once for each parameter.
//saveas will be modified to indicate which parameter is being used each time.

void misalignmentDependence(TCanvas *c1old,
                            Int_t nFiles,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                            TF1 *function,Int_t nParameters,Int_t *parameters,TString *parameternames,TString functionname,
                            Bool_t relative,Bool_t resolution,Bool_t pull,
                            TString saveas)
{
    for (int i = 0; i < nParameters; i++)
    {
        TString saveasi = saveas;
        TString insert = nPart(1,parameternames[i]);
        insert.Prepend(".");
        saveasi.Insert(saveasi.Last('.'),insert);    //insert the parameter name before the file extension
        misalignmentDependence(c1old,
                               nFiles,names,misalignment,values,phases,xvar,yvar,
                               function,parameters[i],parameternames[i],functionname,
                               relative,resolution,pull,
                               saveasi);
    }
}


//This version does not take a canvas as its argument.  It runs trackSplitPlot to produce the canvas.

void misalignmentDependence(Int_t nFiles,TString *files,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                            TF1 *function,Int_t parameter,TString parametername,TString functionname,
                            Bool_t relative,Bool_t resolution,Bool_t pull,
                            TString saveas)
{
    misalignmentDependence(trackSplitPlot(nFiles,files,names,xvar,yvar,relative,resolution,pull,""),
                           nFiles,names,misalignment,values,phases,xvar,yvar,
                           function,parameter,parametername,functionname,
                           relative,resolution,pull,saveas);
}

void misalignmentDependence(Int_t nFiles,TString *files,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                            TF1 *function,Int_t nParameters,Int_t *parameters,TString *parameternames,TString functionname,
                            Bool_t relative,Bool_t resolution,Bool_t pull,
                            TString saveas)
{
    for (int i = 0; i < nParameters; i++)
    {
        TString saveasi = saveas;
        TString insert = nPart(1,parameternames[i]);
        insert.Prepend(".");
        saveasi.Insert(saveasi.Last('.'),insert);    //insert the parameter name before the file extension
        misalignmentDependence(nFiles,files,names,misalignment,values,phases,xvar,yvar,
                               function,parameters[i],parameternames[i],functionname,
                               relative,resolution,pull,
                               saveasi);
    }
}


// This version allows you to use a string for the function.  It creates a TF1 using this string and uses this TF1

void misalignmentDependence(TCanvas *c1old,
                            Int_t nFiles,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                            TString function,Int_t parameter,TString parametername,TString functionname,
                            Bool_t relative,Bool_t resolution,Bool_t pull,
                            TString saveas)
{
    TF1 *f = new TF1("func",function);
    misalignmentDependence(c1old,nFiles,names,misalignment,values,phases,xvar,yvar,f,parameter,parametername,functionname,relative,resolution,pull,saveas);
    delete f;
}

void misalignmentDependence(TCanvas *c1old,
                            Int_t nFiles,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                            TString function,Int_t nParameters,Int_t *parameters,TString *parameternames,TString functionname,
                            Bool_t relative,Bool_t resolution,Bool_t pull,
                            TString saveas)
{
    for (int i = 0; i < nParameters; i++)
    {
        TString saveasi = saveas;
        TString insert = nPart(1,parameternames[i]);
        insert.Prepend(".");
        saveasi.Insert(saveasi.Last('.'),insert);    //insert the parameter name before the file extension
        misalignmentDependence(c1old,
                               nFiles,names,misalignment,values,phases,xvar,yvar,
                               function,parameters[i],parameternames[i],functionname,
                               relative,resolution,pull,
                               saveasi);
    }
}


void misalignmentDependence(Int_t nFiles,TString *files,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                            TString function,Int_t parameter,TString parametername,TString functionname,
                            Bool_t relative,Bool_t resolution,Bool_t pull,
                            TString saveas)
{
    TF1 *f = new TF1("func",function);
    misalignmentDependence(nFiles,files,names,misalignment,values,phases,xvar,yvar,f,parameter,parametername,functionname,relative,resolution,pull,saveas);
    delete f;
}

void misalignmentDependence(Int_t nFiles,TString *files,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                            TString function,Int_t nParameters,Int_t *parameters,TString *parameternames,TString functionname,
                            Bool_t relative,Bool_t resolution,Bool_t pull,
                            TString saveas)
{
    for (int i = 0; i < nParameters; i++)
    {
        TString saveasi = saveas;
        TString insert = nPart(1,parameternames[i]);
        insert.Prepend(".");
        saveasi.Insert(saveasi.Last('.'),insert);    //insert the parameter name before the file extension
        misalignmentDependence(nFiles,files,names,misalignment,values,phases,xvar,yvar,
                               function,parameters[i],parameternames[i],functionname,
                               relative,resolution,pull,
                               saveasi);
    }
}




//This version does not take a function as its argument.  It automatically determines what function, parameter,
//functionname, and parametername to use based on misalignment, xvar, yvar, relative, resolution, and pull.
//However, you have to manually put into the function which plots to fit to what shapes.
//The 2012A data, using the prompt geometry, is a nice example if you want to see an elliptical misalignment.
//If drawfits is true, it draws the fits; otherwise it plots the parameter as a function of misalignment as given by values.

//If the combination of misalignment, xvar, yvar, relative, resolution, pull has a default function to use, it returns true,
// otherwise it returns false.

//This is the version called by makeThesePlots.C

Bool_t misalignmentDependence(TCanvas *c1old,
                              Int_t nFiles,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                              Bool_t drawfits,
                              Bool_t relative,Bool_t resolution,Bool_t pull,
                              TString saveas)
{
    if (xvar == "")
    {
        if (c1old == 0 || misalignment == "" || values == 0) return false;
        misalignmentDependence(c1old,nFiles,names,misalignment,values,phases,xvar,yvar,(TF1*)0,0,"","",relative,resolution,pull,saveas);
        return true;
    }
    TF1 *f = 0;
    TString functionname = "";

    //if only one parameter is of interest
    TString parametername = "";
    Int_t parameter = 9999;

    //if multiple parameters are of interest
    Int_t nParameters = -1;
    TString *parameternames = 0;
    Int_t *parameters = 0;

    if (misalignment == "sagitta")
    {
        if (xvar == "phi" && yvar == "phi" && !resolution && !pull)
        {
            f = new TF1("sine","-[0]*cos([1]*x+[2])");
            f->FixParameter(1,1);
            f->SetParameter(0,6e-4);
            nParameters = 2;
            Int_t tempParameters[2] = {0,2};
            TString tempParameterNames[2] = {"A","B"};
            parameters = tempParameters;
            parameternames = tempParameterNames;
            functionname = "#Delta#phi=-Acos(#phi_{org}+B)";
        }
        if (xvar == "theta" && yvar == "theta" && !resolution && pull)
        {
            f = new TF1("line","-[0]*(x+[1])");
            f->FixParameter(1,-pi/2);
            parametername = "A";
            functionname = "#Delta#theta/#delta(#Delta#theta)=-A(#theta_{org}-#pi/2)";
            parameter = 0;
        }
        if (xvar == "theta" && yvar == "theta" && !resolution && !pull)
        {
            f = new TF1("sine","[0]*sin([1]*x+[2])");
            f->FixParameter(1,2);
            f->FixParameter(2,0);
            parametername = "A";
            functionname = "#Delta#theta=-Asin(2#theta_{org})";
            parameter = 0;
        }
    }
    if (misalignment == "elliptical")
    {
        if (xvar == "phi" && yvar == "dxy" && !resolution && !pull)
        {
            f = new TF1("sine","[0]*sin([1]*x-[2])");
            //f = new TF1("sine","[0]*sin([1]*x-[2]) + [3]");
            f->FixParameter(1,-2);
            f->SetParameter(0,5e-4);

            nParameters = 2;
            Int_t tempParameters[2] = {0,2};
            TString tempParameterNames[2] = {"A;#mum","B"};
            //nParameters = 3;
            //Int_t tempParameters[3] = {0,2,3};
            //TString tempParameterNames[3] = {"A;#mum","B","C;#mum"};

            parameters = tempParameters;
            parameternames = tempParameterNames;
            functionname = "#Deltad_{xy}=-Asin(2#phi_{org}+B)";
            //functionname = "#Deltad_{xy}=-Asin(2#phi_{org}+B)+C";
        }
        if (xvar == "phi" && yvar == "dxy" && !resolution && pull)
        {
            f = new TF1("sine","[0]*sin([1]*x-[2])");
            //f = new TF1("sine","[0]*sin([1]*x-[2]) + [3]");

            f->FixParameter(1,-2);

            nParameters = 2;
            Int_t tempParameters[2] = {0,2};
            TString tempParameterNames[2] = {"A","B"};
            //nParameters = 3;
            //Int_t tempParameters[3] = {0,2,3};
            //TString tempParameterNames[3] = {"A","B","C"};

            parameters = tempParameters;
            parameternames = tempParameterNames;

            functionname = "#Deltad_{xy}/#delta(#Deltad_{xy})=-Asin(2#phi_{org}+B)";
            //functionname = "#Deltad_{xy}/#delta(#Deltad_{xy})=-Asin(2#phi_{org}+B)+C";
        }

        if (xvar == "theta" && yvar == "dz" && !resolution && !pull)
        {
            f = new TF1("line","-[0]*(x-[1])");
            f->FixParameter(1,pi/2);
            parametername = "A;#mum";
            functionname = "#Deltad_{z}=-A(#theta_{org}-#pi/2)";
            parameter = 0;
        }
        /*
        This fit doesn't work
        if (xvar == "theta" && yvar == "dz" && !resolution && pull)
        {
            f = new TF1("sine","[0]*sin([1]*x+[2])");
            f->FixParameter(2,-pi/2);
            f->FixParameter(1,1);
            parametername = "A";
            functionname = "#Deltad_{z}/#delta(#Deltad_{z})=Acos(#theta_{org})";
            parameter = 0;
        }
        */
        if (xvar == "dxy" && yvar == "phi" && !resolution && !pull)
        {
            f = new TF1("line","-[0]*(x-[1])");
            f->FixParameter(1,0);
            parametername = "A;cm^{-1}";
            functionname = "#Delta#phi=-A(d_{xy})_{org}";
            parameter = 0;
        }
        if (xvar == "dxy" && yvar == "phi" && !resolution && pull)
        {
            f = new TF1("line","-[0]*(x-[1])");
            f->FixParameter(1,0);
            parametername = "A;cm^{-1}";
            functionname = "#Delta#phi/#delta(#Delta#phi)=-A(d_{xy})_{org}";
            parameter = 0;
        }
    }
    if (misalignment == "skew")
    {
        if (xvar == "phi" && yvar == "theta" && resolution && !pull)
        {
            f = new TF1("sine","[0]*sin([1]*x+[2])+[3]");
            f->FixParameter(1,2);
            nParameters = 3;
            Int_t tempParameters[3] = {0,2,3};
            TString tempParameterNames[3] = {"A","B","C"};
            parameters = tempParameters;
            parameternames = tempParameterNames;
            functionname = "#sigma(#Delta#theta)=Asin(2#phi_{org}+B)+C";
        }
        if (xvar == "phi" && yvar == "eta" && resolution && !pull)
        {
            f = new TF1("sine","[0]*sin([1]*x+[2])+[3]");
            f->FixParameter(1,2);
            nParameters = 3;
            Int_t tempParameters[3] = {0,2,3};
            TString tempParameterNames[3] = {"A","B","C"};
            parameters = tempParameters;
            parameternames = tempParameterNames;
            functionname = "#sigma(#Delta#eta)=Asin(2#phi_{org}+B)+C";
        }
        if (xvar == "phi" && yvar == "theta" && resolution && pull)
        {
            f = new TF1("sine","[0]*sin([1]*x+[2])+[3]");
            f->FixParameter(1,2);
            nParameters = 3;
            Int_t tempParameters[3] = {0,2,3};
            TString tempParameterNames[3] = {"A","B","C"};
            parameters = tempParameters;
            parameternames = tempParameterNames;
            functionname = "#sigma(#Delta#theta/#delta(#Delta#theta))=Asin(2#phi_{org}+B)+C";
        }
        if (xvar == "phi" && yvar == "eta" && resolution && pull)
        {
            f = new TF1("sine","[0]*sin([1]*x+[2])+[3]");
            f->FixParameter(1,2);
            nParameters = 3;
            Int_t tempParameters[3] = {0,2,3};
            TString tempParameterNames[3] = {"A","B","C"};
            parameters = tempParameters;
            parameternames = tempParameterNames;
            functionname = "#sigma(#Delta#eta/#delta(#Delta#eta))=Asin(2#phi_{org}+B)+C";
        }
        if (xvar == "phi" && yvar == "dz" && !resolution && !pull)
        {
            f = new TF1("tanh","[0]*(tanh([1]*(x+[2]))   )");  // - tanh(([3]-[1])*x+[2]) + 1)");
            //f = new TF1("tanh","[0]*(tanh([1]*(x+[2])) + tanh([1]*([3]-[2]-x)) - 1)");
            f->SetParameter(0,100);
            f->SetParLimits(1,-20,20);
            f->SetParLimits(2,0,pi);
            f->FixParameter(3,pi);
            nParameters = 3;
            Int_t tempParameters[3] = {0,1,2};
            TString tempParameterNames[3] = {"A;#mum","B","C"};
            parameters = tempParameters;
            parameternames = tempParameterNames;
            functionname = "#Deltad_{z}=Atanh(B(#phi_{org}+C))";
            //functionname = "#Deltad_{z}=A(tanh(B(#phi_{org}+C)) + tanh(B(#pi-#phi_{org}-C)) - 1";
        }
    }
    if (misalignment == "layerRot")
    {
        if (xvar == "qoverpt" && yvar == "qoverpt" && !relative && !resolution && !pull)
        {
            f = new TF1("sech","[0]/cosh([1]*(x+[2]))+[3]");
            //f = new TF1("gauss","[0]/exp(([1]*(x+[2]))^2)+[3]");   //sech works better than a gaussian
            f->SetParameter(0,1);
            f->SetParameter(1,1);
            f->SetParLimits(1,0,10);
            f->FixParameter(2,0);
            f->FixParameter(3,0);
            nParameters = 2;
            Int_t tempParameters[2] = {0,1};
            TString tempParameterNames[2] = {"A;e/GeV","B;GeV/e"};
            parameters = tempParameters;
            parameternames = tempParameterNames;
            functionname = "#Delta(q/p_{T})=Asech(B(q/p_{T})_{org})";
        }
    }
    if (misalignment == "telescope")
    {
        if (xvar == "theta" && yvar == "theta" && !relative && !resolution && !pull)
        {
            f = new TF1("gauss","[0]/exp(([1]*(x+[2]))^2)+[3]");
            f->SetParameter(0,1);
            f->SetParameter(1,1);
            f->SetParLimits(1,0,10);
            f->FixParameter(2,-pi/2);
            f->FixParameter(3,0);
            nParameters = 2;
            Int_t tempParameters[2] = {0,1};
            TString tempParameterNames[2] = {"A","B"};
            parameters = tempParameters;
            parameternames = tempParameterNames;
            functionname = "#Delta#theta=Aexp(-(B(#theta_{org}-#pi/2))^{2})";
        }
    }
    if (functionname == "") return false;
    if (drawfits)
    {
        parameter = -parameter-1;
        for (int i = 0; i < nParameters; i++)
            parameters[i] = -parameters[i]-1;
    }
    if (nParameters > 0)
        misalignmentDependence(c1old,nFiles,names,misalignment,values,phases,xvar,yvar,
                               f,nParameters,parameters,parameternames,functionname,relative,resolution,pull,saveas);
    else
        misalignmentDependence(c1old,nFiles,names,misalignment,values,phases,xvar,yvar,
                               f,parameter,parametername,functionname,relative,resolution,pull,saveas);
    delete f;
    return true;

}


//This is the most practically useful version.  It does not take a canvas, but produces it automatically and then determines what
//function to fit it to.

Bool_t misalignmentDependence(Int_t nFiles,TString *files,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                              Bool_t drawfits,
                              Bool_t relative,Bool_t resolution,Bool_t pull,
                              TString saveas)
{
    return misalignmentDependence(trackSplitPlot(nFiles,files,names,xvar,yvar,relative,resolution,pull,""),
                                  nFiles,names,misalignment,values,phases,xvar,yvar,
                                  drawfits,relative,resolution,pull,saveas);
}

Bool_t hasFit(TString misalignment,TString xvar,TString yvar,Bool_t relative,Bool_t resolution,Bool_t pull)
{
    return misalignmentDependence((TCanvas*)0,
                                  0,(TString*)0,misalignment,(Double_t*)0,(Double_t*)0,xvar,yvar,
                                  false,
                                  relative,resolution,pull,
                                  TString(""));
}

//=============
//2. Make Plots
//=============

void makePlots(Int_t nFiles,TString *files,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString directory,
               Bool_t matrix[xsize][ysize])
{
    stufftodelete->SetOwner(true);

    for (Int_t i = 0, totaltime = 0; i < nFiles; i++)
    {
        TFile *f = 0;
        bool exists = false;
        if (files[i] == "") exists = true;

        for (int j = 1; j <= 60*24 && !exists; j++, totaltime++)  //wait up to 1 day for the validation to be finished
        {
            f = TFile::Open(files[i]);
            if (f != 0)
                exists = f->IsOpen();
            delete f;
            if (exists) continue;
            gSystem->Sleep(60000);
            cout << "It's been ";
            if (j >= 60)
                cout << j/60 << " hour";
            if (j >= 120)
                cout << "s";
            if (j % 60 != 0 && j >= 60)
                cout << " and ";
            if (j % 60 != 0)
                cout << j%60 << " minute";
            if (j % 60 >= 2)
                cout << "s";
            cout << endl;
        }
        if (!exists) return;
        if (i == nFiles - 1 && totaltime > nFiles)
            gSystem->Sleep(60000);
    }

    TString directorytomake = directory;
    gSystem->mkdir(directorytomake,true);
    if (misalignment != "")
    {
        directorytomake.Append("/fits");
        gSystem->mkdir(directorytomake);
    }

    for (Int_t x = 0; x < xsize; x++)
    {
        for (Int_t y = 0; y < ysize; y++)
        {
            for (Int_t pull = 0; pull == 0 || (pull == 1 && yvariables[y] != ""); pull++)
            {
                if (false) continue;        //this line is to make it easier to do e.g. all plots involving Delta eta
                                            //(replace false with yvariables[y] != "eta")

                if (!matrix[x][y]) continue;

                if (xvariables[x] == "" && yvariables[y] == "") continue;

                Int_t nPlots = nFiles+4;                     //scatterplot for each (if you uncomment it), profile, resolution, and fits for each.
                vector<TString> s;

                TString slashstring = "";
                if (directory.Last('/') != directory.Length() - 1) slashstring = "/";

                vector<TString> plotnames;
                for (Int_t i = 0; i < nFiles; i++)
                {
                    plotnames.push_back(names[i]);   //this is plotnames[i]
                    plotnames[i].ReplaceAll(" ","");
                }

                plotnames.push_back("");             //this is plotnames[nFiles], but gets changed
                if (yvariables[y] == "")
                    plotnames[nFiles] = "orghist";
                else if (xvariables[x] == "")
                    plotnames[nFiles] = "hist";
                else
                    plotnames[nFiles] = "profile";

                plotnames.push_back("resolution");   //this is plotnames[nFiles+1]

                plotnames.push_back("");             //this is plotnames[nFiles+2]
                plotnames.push_back("");             //this is plotnames[nFiles+3]
                if (plotnames[nFiles] == "profile")
                {
                    plotnames[nFiles+2] = ".profile";
                    plotnames[nFiles+2].Prepend(misalignment);
                    plotnames[nFiles+3] = ".resolution";
                    plotnames[nFiles+3].Prepend(misalignment);
                    plotnames[nFiles+2].Prepend("fits/");
                    plotnames[nFiles+3].Prepend("fits/");
                }
                else
                {
                    plotnames[nFiles+2] = "profile.";
                    plotnames[nFiles+2].Append(misalignment);
                    plotnames[nFiles+3] = "resolution.";
                    plotnames[nFiles+3].Append(misalignment);
                }

                TString pullstring = "";
                if (pull) pullstring = "pull.";

                TString xvarstring = xvariables[x];
                if (xvariables[x] != "runNumber" && xvariables[x] != "nHits" && xvariables[x] != "") xvarstring.Append("_org");
                if (xvariables[x] != "" && yvariables[y] != "") xvarstring.Append(".");

                TString yvarstring = yvariables[y];
                if (yvariables[y] != "") yvarstring.Prepend("Delta_");

                TString relativestring = "";
                if (relativearray[y]) relativestring = ".relative";

                for (Int_t i = 0; i < nPlots; i++)
                {
                    stringstream ss;
                    ss << directory << slashstring << plotnames[i] << "." << pullstring
                       << xvarstring << yvarstring << relativestring << ".pngepsroot";
                    s.push_back(ss.str());
                    if (misalignment != "")
                    {
                        TString wrongway = misalignment;
                        TString rightway = misalignment;
                        wrongway.Append (".pull");
                        rightway.Prepend("pull.");
                        s[i].ReplaceAll(wrongway,rightway);
                    }
                }

                Int_t i;
                for (i = 0; i < nFiles; i++)
                {
                    if (xvariables[x] == "" || yvariables[y] == "") continue;
                    //uncomment this section to make scatterplots
                    /*
                    trackSplitPlot(files[i],xvariables[x],yvariables[y],false,relativearray[y],false,(bool)pull,s[i]);
                    stufftodelete->Clear();
                    for ( ; gROOT->GetListOfCanvases()->GetEntries() > 0; )
                        deleteCanvas( gROOT->GetListOfCanvases()->Last());
                    for ( ; gROOT->GetListOfFiles()->GetEntries() > 0; )
                        delete (TFile*)gROOT->GetListOfFiles()->Last();
                    */
                }

                if (xvariables[x] != "" && yvariables[y] != "")
                {
                    //make profile
                    TCanvas *c1 = trackSplitPlot(nFiles,files,names,xvariables[x],yvariables[y],relativearray[y],false,(bool)pull,s[i]);
                    if (misalignmentDependence(c1,nFiles,names,misalignment,values,phases,xvariables[x],yvariables[y],
                                               true,relativearray[y],false,(bool)pull,s[i+2]))
                    {
                        s[i+2].ReplaceAll(".png",".parameter.png");
                        misalignmentDependence(c1,nFiles,names,misalignment,values,phases,xvariables[x],yvariables[y],
                                                   false,relativearray[y],false,(bool)pull,s[i+2]);
                    }
                    stufftodelete->Clear();
                    for ( ; gROOT->GetListOfCanvases()->GetEntries() > 0; )
                        deleteCanvas( gROOT->GetListOfCanvases()->Last());
                    for ( ; gROOT->GetListOfFiles()->GetEntries() > 0; )
                        delete (TFile*)gROOT->GetListOfFiles()->Last();

                    //make resolution plot
                    TCanvas *c2 = trackSplitPlot(nFiles,files,names,xvariables[x],yvariables[y],relativearray[y],true ,(bool)pull,s[i+1]);
                    if (misalignmentDependence(c2,nFiles,names,misalignment,values,phases,xvariables[x],yvariables[y],
                                               true,relativearray[y],true,(bool)pull,s[i+3]))
                    {
                        s[i+3].ReplaceAll(".png",".parameter.png");
                        misalignmentDependence(c2,nFiles,names,misalignment,values,phases,xvariables[x],yvariables[y],
                                                   false,relativearray[y],true,(bool)pull,s[i+3]);
                    }
                    stufftodelete->Clear();
                    for ( ; gROOT->GetListOfCanvases()->GetEntries() > 0; )
                        deleteCanvas( gROOT->GetListOfCanvases()->Last());
                    for ( ; gROOT->GetListOfFiles()->GetEntries() > 0; )
                        delete (TFile*)gROOT->GetListOfFiles()->Last();
                }
                else
                {
                    //make histogram
                    TCanvas *c1 = trackSplitPlot(nFiles,files,names,xvariables[x],yvariables[y],relativearray[y],false,(bool)pull,s[i]);
                    if (misalignmentDependence(c1,nFiles,names,misalignment,values,phases,xvariables[x],yvariables[y],
                                               true,relativearray[y],false,(bool)pull,s[i+2]))
                    {
                        misalignmentDependence(c1,nFiles,names,misalignment,values,phases,xvariables[x],yvariables[y],
                                               true,relativearray[y],true,(bool)pull,s[i+3]);
                    }
                    stufftodelete->Clear();
                    for ( ; gROOT->GetListOfCanvases()->GetEntries() > 0; )
                        deleteCanvas( gROOT->GetListOfCanvases()->Last());
                    for ( ; gROOT->GetListOfFiles()->GetEntries() > 0; )
                        delete (TFile*)gROOT->GetListOfFiles()->Last();
                }
            }
            cout << y + ysize * x + 1 << "/" << xsize*ysize << endl;
        }
    }
}

void makePlots(Int_t nFiles,TString *files,TString *names,TString directory, Bool_t matrix[xsize][ysize])
{
    makePlots(nFiles,files,names,"",(Double_t*)0,(Double_t*)0,directory,
              matrix);
}

void makePlots(TString file,TString misalignment,Double_t *values,Double_t *phases,TString directory,Bool_t matrix[xsize][ysize])
{
    int n = file.CountChar(',') + 1;
    TString *files = new TString[n];
    TString *names = new TString[n];
    setTDRStyle();
    vector<Color_t> tempcolors = colors;
    vector<Style_t> tempstyles = styles;
    for (int i = 0; i < n; i++)
    {
        TString thisfile = nPart(i+1,file,",");
        int numberofpipes = thisfile.CountChar('|');
        if (numberofpipes >= 0 && nPart(numberofpipes+1,thisfile,"|").IsDigit())
        {
            if (numberofpipes >= 1 && nPart(numberofpipes,thisfile,"|").IsDigit())
            {
                colors[i] = nPart(numberofpipes,thisfile,"|").Atoi();
                styles[i] = nPart(numberofpipes+1,thisfile,"|").Atoi();
                thisfile.Remove(thisfile.Length() - nPart(numberofpipes,thisfile,"|").Length() - nPart(numberofpipes+1,thisfile,"|").Length() - 2);
            }
            else
            {
                colors[i] = nPart(numberofpipes + 1,thisfile,"|").Atoi();
                thisfile.Remove(thisfile.Length() - nPart(numberofpipes+1,thisfile,"|").Length() - 2);
            }
        }
        files[i] = nPart(1,thisfile,"=",true);
        names[i] = nPart(2,thisfile,"=",false);
    }
    if (n == 1 && names[0] == "")
        names[0] = "scatterplot";     //With 1 file there's no legend, so this is only used in the filename of the scatterplots, if made
    makePlots(n,files,names,misalignment,values,phases,directory,matrix);
    delete[] files;
    delete[] names;
    colors = tempcolors;
    styles = tempstyles;
}

void makePlots(TString file,TString directory,Bool_t matrix[xsize][ysize])
{
    makePlots(file,"",(Double_t*)0,(Double_t*)0,directory,matrix);
}

//***************************************************************************
//functions to make plots for 1 row, column, or cell of the matrix
//examples:
//   xvar = "nHits", yvar = "ptrel" - makes plots of nHits vs Delta_pt/pt_org
//   xvar = "all",   yvar = "pt"    - makes all plots involving Delta_pt
//                                    (not Delta_pt/pt_org)
//   xvar = "",      yvar = "all"   - makes all histograms of Delta_???
//                                    (including Delta_pt/pt_org)
//***************************************************************************

void makePlots(Int_t nFiles,TString *files,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString directory,
               TString xvar,TString yvar)
{
    Bool_t matrix[xsize][ysize];
    for (int x = 0; x < xsize; x++)
        for (int y = 0; y < ysize; y++)
        {
            bool xmatch = (xvar == "all" || xvar == xvariables[x]);
            bool ymatch = (yvar == "all" || yvar == yvariables[y]);
            if (yvar == "pt" && yvariables[y] == "pt" && relativearray[y] == true)
                ymatch = false;
            if (yvar == "ptrel" && yvariables[y] == "pt" && relativearray[y] == true)
                ymatch = true;
            matrix[x][y] = (xmatch && ymatch);
        }
    makePlots(nFiles,files,names,misalignment,values,phases,directory,matrix);
}

void makePlots(Int_t nFiles,TString *files,TString *names,TString directory,
               TString xvar,TString yvar)
{
    makePlots(nFiles,files,names,"",(Double_t*)0,(Double_t*)0,directory,
              xvar,yvar);
}

void makePlots(TString file,TString misalignment,Double_t *values,Double_t *phases,TString directory,
               TString xvar,TString yvar)
{
    int n = file.CountChar(',') + 1;
    TString *files = new TString[n];
    TString *names = new TString[n];
    setTDRStyle();
    vector<Color_t> tempcolors = colors;
    vector<Style_t> tempstyles = styles;
    for (int i = 0; i < n; i++)
    {
        TString thisfile = nPart(i+1,file,",");
        int numberofpipes = thisfile.CountChar('|');
        if (numberofpipes >= 0 && nPart(numberofpipes+1,thisfile,"|").IsDigit())
        {
            if (numberofpipes >= 1 && nPart(numberofpipes,thisfile,"|").IsDigit())
            {
                colors[i] = nPart(numberofpipes,thisfile,"|").Atoi();
                styles[i] = nPart(numberofpipes+1,thisfile,"|").Atoi();
                thisfile.Remove(thisfile.Length() - nPart(numberofpipes,thisfile,"|").Length() - nPart(numberofpipes+1,thisfile,"|").Length() - 2);
            }
            else
            {
                colors[i] = nPart(numberofpipes + 1,thisfile,"|").Atoi();
                thisfile.Remove(thisfile.Length() - nPart(numberofpipes+1,thisfile,"|").Length() - 2);
            }
        }
        files[i] = nPart(1,thisfile,"=",true);
        names[i] = nPart(2,thisfile,"=",false);
    }
    if (n == 1 && names[0] == "")
        names[0] = "scatterplot";     //With 1 file there's no legend, so this is only used in the filename of the scatterplots, if made
    makePlots(n,files,names,misalignment,values,phases,directory,xvar,yvar);
    delete[] files;
    delete[] names;
    colors = tempcolors;
    styles = tempstyles;
}

void makePlots(TString file,TString directory,TString xvar,TString yvar)
{
    makePlots(file,"",(Double_t*)0,(Double_t*)0,directory,xvar,yvar);
}

//***************************
//functions to make all plots
//***************************

void makePlots(Int_t nFiles,TString *files,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString directory)
{
    makePlots(nFiles,files,names,misalignment,values,phases,directory,"all","all");
}

void makePlots(Int_t nFiles,TString *files,TString *names,TString directory)
{
    makePlots(nFiles,files,names,"",(Double_t*)0,(Double_t*)0,directory);
}

void makePlots(TString file,TString misalignment,Double_t *values,Double_t *phases,TString directory)
{
    int n = file.CountChar(',') + 1;
    TString *files = new TString[n];
    TString *names = new TString[n];
    setTDRStyle();
    vector<Color_t> tempcolors = colors;
    vector<Style_t> tempstyles = styles;
    for (int i = 0; i < n; i++)
    {
        TString thisfile = nPart(i+1,file,",");
        int numberofpipes = thisfile.CountChar('|');
        if (numberofpipes >= 0 && nPart(numberofpipes+1,thisfile,"|").IsDigit())
        {
            if (numberofpipes >= 1 && nPart(numberofpipes,thisfile,"|").IsDigit())
            {
                colors[i] = nPart(numberofpipes,thisfile,"|").Atoi();
                styles[i] = nPart(numberofpipes+1,thisfile,"|").Atoi();
                thisfile.Remove(thisfile.Length() - nPart(numberofpipes,thisfile,"|").Length() - nPart(numberofpipes+1,thisfile,"|").Length() - 2);
            }
            else
            {
                colors[i] = nPart(numberofpipes + 1,thisfile,"|").Atoi();
                thisfile.Remove(thisfile.Length() - nPart(numberofpipes+1,thisfile,"|").Length() - 2);
            }
        }
        files[i] = nPart(1,thisfile,"=",true);
        names[i] = nPart(2,thisfile,"=",false);
    }
    if (n == 1 && names[0] == "")
        names[0] = "scatterplot";     //With 1 file there's no legend, so this is only used in the filename of the scatterplots, if made
    makePlots(n,files,names,misalignment,values,phases,directory);
    delete[] files;
    delete[] names;
    colors = tempcolors;
    styles = tempstyles;
}

void makePlots(TString file,TString directory)
{
    makePlots(file,"",(Double_t*)0,(Double_t*)0,directory);
}

//=============
//3. Axis Label
//=============

TString fancyname(TString variable)
{
    if (variable == "pt")
        return "p_{T}";
    else if (variable == "phi")
        return "#phi";
    else if (variable == "eta")
        return "#eta";
    else if (variable == "theta")
        return "#theta";
    else if (variable == "qoverpt")
        return "(q/p_{T})";
    else if (variable == "runNumber")
        return "run number";
    else if (variable == "dxy" || variable == "dz")
        return variable.ReplaceAll("d","d_{").Append("}");
    else
        return variable;
}

//this gives the units, to be put in the axis label
TString units(TString variable,Char_t axis)
{
    if (variable == "pt")
        return "GeV";
    if (variable == "dxy" || variable == "dz")
    {
        if (axis == 'y')
            return "#mum";      //in the tree, it's listed in centimeters, but in trackSplitPlot the value is divided by 1e4
        if (axis == 'x')
            return "cm";
    }
    if (variable == "qoverpt")
        return "e/GeV";
    return "";
}


//this gives the full axis label, including units.  It can handle any combination of relative, resolution, and pull.
TString axislabel(TString variable, Char_t axis, Bool_t relative, Bool_t resolution, Bool_t pull)
{
    stringstream s;
    if (resolution && axis == 'y')
        s << "#sigma(";
    if (axis == 'y')
        s << "#Delta";
    s << fancyname(variable);
    if (relative && axis == 'y')
    {
        s << " / ";
        if (!pull)
            s << "(";
        s << fancyname(variable);
    }
    Bool_t nHits = (variable[0] == 'n' && variable[1] == 'H' && variable[2] == 'i'
                                       && variable[3] == 't' && variable[4] == 's');
    if (relative || (axis == 'x' && variable != "runNumber" && !nHits))
        s << "_{org}";
    if (axis == 'y')
    {
        if (pull)
        {
            s << " / #delta(#Delta" << fancyname(variable);
            if (relative)
                s << " / " << fancyname(variable) << "_{org}";
            s << ")";
        }
        else
        {
            if (!relative)
                s << " / ";
            s << "#sqrt{2}";
            if (relative)
                s << ")";
        }
    }
    if (resolution && axis == 'y')
        s << ")";
    if (((!relative && !pull) || axis == 'x') && units(variable,axis) != "")
        s << " (" << units(variable,axis) << ")";
    TString result = s.str();
    result.ReplaceAll("d_{xy}_{org}","(d_{xy})_{org}").ReplaceAll("d_{z}_{org}","(d_{z})_{org}").ReplaceAll("p_{T}_{org}","(p_{T})_{org}");
    return result;
}

void setAxisLabels(TH1 *p, PlotType type,TString xvar,TString yvar,Bool_t relative,Bool_t pull)
{
    if (type == Histogram)
        p->SetXTitle(axislabel(yvar,'y',relative,false,pull));
    if (type == ScatterPlot || type == Profile || type == Resolution || type == OrgHistogram)
        p->SetXTitle(axislabel(xvar,'x'));

    if (type == ScatterPlot || type == Profile)
        p->SetYTitle(axislabel(yvar,'y',relative,false,pull));
    if (type == Resolution)
        p->SetYTitle(axislabel(yvar,'y',relative,true,pull));
}

void setAxisLabels(TMultiGraph *p, PlotType type,TString xvar,TString yvar,Bool_t relative,Bool_t pull)
{
    if (type == Histogram)
        p->GetXaxis()->SetTitle(axislabel(yvar,'y',relative,false,pull));
    if (type == ScatterPlot || type == Profile || type == Resolution || type == OrgHistogram)
        p->GetXaxis()->SetTitle(axislabel(xvar,'x'));

    if (type == ScatterPlot || type == Profile)
        p->GetYaxis()->SetTitle(axislabel(yvar,'y',relative,false,pull));
    if (type == Resolution)
        p->GetYaxis()->SetTitle(axislabel(yvar,'y',relative,true,pull));
}


TString nPart(Int_t part,TString string,TString delimit,Bool_t removerest)
{
    if (part <= 0) return "";
    for (int i = 1; i < part; i++)    //part-1 times
    {
        if (string.Index(delimit) < 0) return "";
        string.Replace(0,string.Index(delimit)+1,"",0);
    }
    if (string.Index(delimit) >= 0 && removerest)
        string.Remove(string.Index(delimit));
    return string;
}

//==============
//4. Axis Limits
//==============


Double_t findStatistic(Statistic what,Int_t nFiles,TString *files,TString var,Char_t axis,Bool_t relative,Bool_t pull)
{
    Double_t x = 0,              //if axis == 'x', var_org goes in x; if axis == 'y', Delta_var goes in x
             rel = 1,            //if relative, var_org goes in rel.  x is divided by rel, so you get Delta_var/var_org
             sigma1 = 1,         //if pull, the error for split track 1 goes in sigma1 and the error for split track 2 goes in sigma2.
             sigma2 = 1,         //x is divided by sqrt(sigma1^2+sigma2^2).  If !pull && axis == 'y', this divides by sqrt(2)
             sigmaorg = 0;       // because we want the error in one track.  sigmaorg is used when relative && pull
    Int_t xint = 0, xint2 = 0;   //xint is used for run number and nHits.  xint2 is used for nhits because each event has 2 values.

    Int_t runNumber = 0;         //this is used to make sure the run number is between minrun and maxrun

    if (axis == 'x')
    {
        sigma1 = 1/sqrt(2);      //if axis == 'x' don't divide by sqrt(2)
        sigma2 = 1/sqrt(2);
    }

    Double_t totallength = 0;
    Double_t result = 0;
    if (what == Minimum) result = 1e100;
    if (what == Maximum) result = -1e100;

    Double_t average = 0;
    if (what == RMS)
        average = findStatistic(Average,nFiles,files,var,axis,relative,pull);

    Bool_t nHits = (var[0] == 'n' && var[1] == 'H' && var[2] == 'i'    //includes nHits, nHitsTIB, etc.
                                  && var[3] == 't' && var[4] == 's');

    stringstream sx,srel,ssigma1,ssigma2,ssigmaorg;

    if (axis == 'y')
        sx << "Delta_";
    sx << var;
    if (axis == 'x' && var != "runNumber" && !nHits)
        sx << "_org";
    if (axis == 'x' && nHits)
        sx << "1_spl";
    TString variable = sx.str(),
            variable2 = variable;
    variable2.ReplaceAll("1_spl","2_spl");

    TString relvariable = "1";
    if (relative)
    {
        srel << var << "_org";
        relvariable = srel.str();
    }

    if (pull)
    {
        ssigma1 << var << "1Err_spl";
        ssigma2 << var << "2Err_spl";
    }
    TString sigma1variable = ssigma1.str();
    TString sigma2variable = ssigma2.str();

    if (pull && relative)
        ssigmaorg << var << "Err_org";
    TString sigmaorgvariable = ssigmaorg.str();

    if (!relative && !pull && (variable == "Delta_dxy" || variable == "Delta_dz"))
        rel = 1e-4;                                           //it's in cm but we want um

    for (Int_t j = 0; j < nFiles; j++)
    {
        if (((var == "runNumber" && what != Maximum) ? findMax(files[j],"runNumber",'x') < 2 : false) || files[j] == "")  //if it's MC data (run 1), the run number is meaningless
            continue;
        TFile *f = TFile::Open(files[j]);
        TTree *tree = (TTree*)f->Get("cosmicValidation/splitterTree");
        if (tree == 0)
            tree = (TTree*)f->Get("splitterTree");
        Int_t length = tree->GetEntries();

        tree->SetBranchAddress("runNumber",&runNumber);
        if (var == "runNumber")
            tree->SetBranchAddress(variable,&xint);
        else if (nHits)
        {
            tree->SetBranchAddress(variable,&xint);
            tree->SetBranchAddress(variable2,&xint2);
        }
        else
            tree->SetBranchAddress(variable,&x);

        if (relative)
            tree->SetBranchAddress(relvariable,&rel);
        if (pull)
        {
            tree->SetBranchAddress(sigma1variable,&sigma1);
            tree->SetBranchAddress(sigma2variable,&sigma2);
        }
        if (relative && pull)
            tree->SetBranchAddress(sigmaorgvariable,&sigmaorg);

        for (Int_t i = 0; i<length; i++)
        {
            tree->GetEntry(i);
            if (var == "runNumber" || nHits)
                x = xint;
            if (var == "runNumber")
                runNumber = x;
            if (var == "phi" && x >= pi)
                x -= 2*pi;
            if (var == "phi" && x <= -pi)
                x += 2*pi;
            if ((runNumber < minrun && runNumber > 1) || (runNumber > maxrun && maxrun > 0)) continue;

            totallength++;

            Double_t error;
            if (relative && pull)
                error = sqrt((sigma1/rel)*(sigma1/rel) + (sigma2/rel)*(sigma2/rel) + (sigmaorg*x/(rel*rel))*(sigmaorg*x/(rel*rel)));
            else
                error = sqrt(sigma1 * sigma1 + sigma2 * sigma2);   // = 1 if axis == 'x' && !pull
                                                                   // = sqrt(2) if axis == 'y' && !pull, so that you get the error in 1 track
                                                                   //       when you divide by it
            x /= (rel * error);
            if (!std::isfinite(x))  //e.g. in data with no pixels, the error occasionally comes out to be NaN
                continue;           //Filling a histogram with NaN is irrelevant, but here it would cause the whole result to be NaN

            if (what == Minimum && x < result)
                result = x;
            if (what == Maximum && x > result)
                result = x;
            if (what == Average)
                result += x;
            if (what == RMS)
                result += (x - average) * (x - average);
            if (nHits)
            {
                x = xint2;
                if (what == Minimum && x < result)
                    result = x;
                if (what == Maximum && x > result)
                    result = x;
                if (what == Average)
                    result += x;
                if (what == RMS)
                    result += (x - average) * (x - average);
            }
        }
        delete f;         //automatically closes the file
    }
    if (nHits) totallength *= 2;
    if (what == Average) result /= totallength;
    if (what == RMS)  result = sqrt(result / (totallength - 1));
    return result;
}

Double_t findAverage(Int_t nFiles,TString *files,TString var,Char_t axis,Bool_t relative,Bool_t pull)
{
    return findStatistic(Average,nFiles,files,var,axis,relative,pull);
}

Double_t findMin(Int_t nFiles,TString *files,TString var,Char_t axis,Bool_t relative,Bool_t pull)
{
    return findStatistic(Minimum,nFiles,files,var,axis,relative,pull);
}

Double_t findMax(Int_t nFiles,TString *files,TString var,Char_t axis,Bool_t relative,Bool_t pull)
{
    return findStatistic(Maximum,nFiles,files,var,axis,relative,pull);
}

Double_t findRMS(Int_t nFiles,TString *files,TString var,Char_t axis,Bool_t relative,Bool_t pull)
{
    return findStatistic(RMS,nFiles,files,var,axis,relative,pull);
}


//These functions are for 1 file

Double_t findStatistic(Statistic what,TString file,TString var,Char_t axis,Bool_t relative,Bool_t pull)
{
    return findStatistic(what,1,&file,var,axis,relative,pull);
}

Double_t findAverage(TString file,TString var,Char_t axis,Bool_t relative,Bool_t pull)
{
    return findStatistic(Average,file,var,axis,relative,pull);
}

Double_t findMin(TString file,TString var,Char_t axis,Bool_t relative,Bool_t pull)
{
    return findStatistic(Minimum,file,var,axis,relative,pull);
}

Double_t findMax(TString file,TString var,Char_t axis,Bool_t relative,Bool_t pull)
{
    return findStatistic(Maximum,file,var,axis,relative,pull);
}

Double_t findRMS(TString file,TString var,Char_t axis,Bool_t relative,Bool_t pull)
{
    return findStatistic(RMS,file,var,axis,relative,pull);
}




//This puts the axis limits that should be used for trackSplitPlot in min and max.
//Default axis limits are defined for pt, qoverpt, dxy, dz, theta, eta, and phi.
//For run number and nHits, the minimum and maximum are used.
//For any other variable, average +/- 5*rms are used.
//To use this instead of the default values, just comment out the part that says [else] if (var == "?") {min = ?; max = ?;}

void axislimits(Int_t nFiles,TString *files,TString var,Char_t axis,Bool_t relative,Bool_t pull,Double_t &min,Double_t &max)
{
    bool pixel = subdetector.Contains("PIX");
    if (axis == 'x')
    {
        Bool_t nHits = (var[0] == 'n' && var[1] == 'H' && var[2] == 'i'
                                      && var[3] == 't' && var[4] == 's');
        if (var == "pt")
        {
            min = 5;
            max = 100;
        }
        else if (var == "qoverpt")
        {
            min = -.35;
            max = .35;
        }
        else if (var == "dxy")
        {
            min = -100;
            max = 100;
            if (pixel)
            {
                min = -10;
                max = 10;
            }
        }
        else if (var == "dz")
        {
            min = -250;
            max = 250;
            if (pixel)
            {
                min = -25;
                max = 25;
            }
        }
        else if (var == "theta")
        {
            min = .5;
            max = 2.5;
        }
        else if (var == "eta")
        {
            min = -1.2;
            max = 1.2;
        }
        else if (var == "phi")
        {
            min = -3;
            max = 0;
        }
        else if (var == "runNumber" || nHits)
        {
            min = findMin(nFiles,files,var,'x') - .5;
            max = findMax(nFiles,files,var,'x') + .5;
        }
        else
        {
            cout << "No x axis limits for " << var << ".  Using average +/- 5*rms" << endl;
            Double_t average = findAverage(nFiles,files,var,'x');
            Double_t rms = findRMS (nFiles,files,var,'x');
            max = TMath::Min(average + 5 * rms,findMax(nFiles,files,var,'x'));
            min = TMath::Max(average - 5 * rms,findMin(nFiles,files,var,'x'));
        }
    }
    if (axis == 'y')
    {
        if (pull)
        {
            min = -5;
            max = 5;
        }
        else if (var == "pt" && relative)
        {
            min = -.06;
            max = .06;
        }
        else if (var == "pt" && !relative)
        {
            min = -.8;
            max = .8;
        }
        else if (var == "qoverpt")
        {
            min = -.0025;
            max = .0025;
        }
        else if (var == "dxy")
        {
            min = -1250;
            max = 1250;
            if (pixel)
            {
                min = -125;
                max = 125;
            }
        }
        else if (var == "dz")
        {
            min = -2000;
            max = 2000;
            if (pixel)
            {
                min = -200;
                max = 200;
            }
        }
        else if (var == "theta")
        {
            min = -.01;
            max = .01;
            if (pixel)
            {
                min = -.005;
                max = .005;
            }
        }
        else if (var == "eta")
        {
            min = -.007;
            max = .007;
            if (pixel)
            {
                min = -.003;
                max = .003;
            }
        }
        else if (var == "phi")
        {
            min = -.002;
            max = .002;
        }
        else
        {
            cout << "No y axis limits for " << var << ".  Using average +/- 5 * rms." << endl;
            Double_t average = 0 /*findAverage(nFiles,files,var,'y',relative,pull)*/;
            Double_t rms = findRMS (nFiles,files,var,'y',relative,pull);
            min = TMath::Max(TMath::Max(-TMath::Abs(average) - 5*rms,
                             findMin(nFiles,files,var,'y',relative,pull)),
                             -findMax(nFiles,files,var,'y',relative,pull));
            max = -min;
        }
    }
}

//===============
//5. Place Legend
//===============

Double_t placeLegend(TLegend *l, Double_t width, Double_t height, Double_t x1min, Double_t y1min, Double_t x2max, Double_t y2max)
{
    for (int i = legendGrid; i >= 0; i--)
    {
        for (int j = legendGrid; j >= 0; j--)
        {
            Double_t x1 = x1min * (1-(double)i/legendGrid) + (x2max - width)  * (double)i/legendGrid - margin*width;
            Double_t y1 = y1min * (1-(double)j/legendGrid) + (y2max - height) * (double)j/legendGrid - margin*height;
            Double_t x2 = x1 + (1+2*margin) * width;
            Double_t y2 = y1 + (1+2*margin) * height;
            if (fitsHere(l,x1,y1,x2,y2))
            {
                x1 += margin*width;
                y1 += margin*height;
                x2 -= margin*width;
                y2 -= margin*height;
                l->SetX1(x1);
                l->SetY1(y1);
                l->SetX2(x2);
                l->SetY2(y2);
                return y2max;
            }
        }
    }
    Double_t newy2max = y2max + increaseby * (y2max-y1min);
    Double_t newheight = height * (newy2max - y1min) / (y2max - y1min);
    return placeLegend(l,width,newheight,x1min,y1min,x2max,newy2max);
}

Bool_t fitsHere(TLegend *l,Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
    Bool_t fits = true;
    TList *list = l->GetListOfPrimitives();
    for (Int_t k = 0; list->At(k) != 0 && fits; k++)
    {
        TObject *obj = ((TLegendEntry*)(list->At(k)))->GetObject();
        if (obj == 0) continue;
        TClass *cl = obj->IsA();

        //Histogram, drawn as a histogram
        if (cl->InheritsFrom("TH1") && !cl->InheritsFrom("TH2") && !cl->InheritsFrom("TH3")
         && cl != TProfile::Class() && ((TH1*)obj)->GetMarkerColor() == kWhite)
        {
            Int_t where = 0;
            TH1 *h = (TH1*)obj;
            for (Int_t i = 1; i <= h->GetNbinsX() && fits; i++)
            {
                if (h->GetBinLowEdge(i) + h->GetBinWidth(i) < x1) continue;   //to the left of the legend
                if (h->GetBinLowEdge(i)                     > x2) continue;   //to the right of the legend
                if (h->GetBinContent(i) > y1 && h->GetBinContent(i) < y2) fits = false;   //inside the legend
                if (h->GetBinContent(i) < y1)
                {
                    if (where == 0) where = -1;             //below the legend
                    if (where == 1) fits = false;           //a previous bin was above it so there's a vertical line through it
                }
                if (h->GetBinContent(i) > y2)
                {
                    if (where == 0) where = 1;              //above the legend
                    if (where == -1) fits = false;          //a previous bin was below it so there's a vertical line through it
                }
            }
            continue;
        }
        //Histogram, drawn with Draw("P")
        else if (cl->InheritsFrom("TH1") && !cl->InheritsFrom("TH2") && !cl->InheritsFrom("TH3")
              && cl != TProfile::Class())
        //Probably TProfile would be the same but I haven't tested it
        {
            TH1 *h = (TH1*)obj;
            for (Int_t i = 1; i <= h->GetNbinsX() && fits; i++)
            {
                if (h->GetBinLowEdge(i) + h->GetBinWidth(i)/2 < x1) continue;
                if (h->GetBinLowEdge(i)                       > x2) continue;
                if (h->GetBinContent(i) > y1 && h->GetBinContent(i) < y2) fits = false;
                if (h->GetBinContent(i) + h->GetBinError(i) > y2 && h->GetBinContent(i) - h->GetBinError(i) < y2) fits = false;
                if (h->GetBinContent(i) + h->GetBinError(i) > y1 && h->GetBinContent(i) - h->GetBinError(i) < y1) fits = false;
            }
        }
        else if (cl->InheritsFrom("TF1") && !cl->InheritsFrom("TF2"))
        {
            TF1 *f = (TF1*)obj;
            Double_t max = f->GetMaximum(x1,x2);
            Double_t min = f->GetMinimum(x1,x2);
            if (min < y2 && max > y1) fits = false;
        }
        // else if (cl->InheritsFrom(...... add more objects here
        else
        {
            cout << "Don't know how to place the legend around objects of type " << obj->ClassName() << "." << endl
                 << "Add this class into fitsHere() if you want it to work properly." << endl
                 << "The legend will still be placed around any other objects." << endl;
        }
    }
    return fits;
}

//============
//6. TDR Style
//============

void setTDRStyle() {

  if (styleset) return;
  styleset = true;
  // For the canvas:
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetCanvasDefH(600); //Height of canvas
  gStyle->SetCanvasDefW(600); //Width of canvas
  gStyle->SetCanvasDefX(0);   //POsition on screen
  gStyle->SetCanvasDefY(0);

  // For the Pad:
  gStyle->SetPadBorderMode(1);
  // gStyle->SetPadBorderSize(Width_t size = 1);
  gStyle->SetPadColor(kWhite);
  gStyle->SetPadGridX(false);
  gStyle->SetPadGridY(false);
  gStyle->SetGridColor(0);
  gStyle->SetGridStyle(3);
  gStyle->SetGridWidth(1);

  // For the frame:
  gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameBorderSize(1);
  gStyle->SetFrameFillColor(0);
  gStyle->SetFrameFillStyle(0);
  gStyle->SetFrameLineColor(1);
  gStyle->SetFrameLineStyle(1);
  gStyle->SetFrameLineWidth(1);

  // For the histo:
  // gStyle->SetHistFillColor(1);
  // gStyle->SetHistFillStyle(0);
  gStyle->SetHistLineColor(1);
  gStyle->SetHistLineStyle(0);
  //gStyle->SetHistLineWidth(1);
  // gStyle->SetLegoInnerR(Float_t rad = 0.5);
  // gStyle->SetNumberContours(Int_t number = 20);

  gStyle->SetEndErrorSize(2);
  //gStyle->SetErrorMarker(20);
  gStyle->SetErrorX(0.);

  gStyle->SetMarkerStyle(7);

  //For the fit/function:
  gStyle->SetOptFit(1);
  gStyle->SetFitFormat("5.4g");
  gStyle->SetFuncColor(2);
  gStyle->SetFuncStyle(1);
  gStyle->SetFuncWidth(1);

  //For the date:
  gStyle->SetOptDate(0);
  // gStyle->SetDateX(Float_t x = 0.01);
  // gStyle->SetDateY(Float_t y = 0.01);


  /*
  // For the statistics box:
  gStyle->SetOptFile(0);
  gStyle->SetOptStat("mr");
  gStyle->SetStatColor(kWhite);
  gStyle->SetStatFont(42);
  gStyle->SetStatFontSize(0.04);///---> gStyle->SetStatFontSize(0.025);
  gStyle->SetStatTextColor(1);
  gStyle->SetStatFormat("6.4g");
  gStyle->SetStatBorderSize(1);
  gStyle->SetStatH(0.1);
  gStyle->SetStatW(0.2);///---> gStyle->SetStatW(0.15);
  */

  // gStyle->SetStatStyle(Style_t style = 1001);
  // gStyle->SetStatX(Float_t x = 0);
  // gStyle->SetStatY(Float_t y = 0);

  // Margins:
  gStyle->SetPadTopMargin(0.05);
  gStyle->SetPadBottomMargin(0.13);
  gStyle->SetPadLeftMargin(0.16);
  gStyle->SetPadRightMargin(0.04);

  // For the Global title:

  gStyle->SetOptTitle(0);
  gStyle->SetTitleFont(42);
  gStyle->SetTitleColor(1);
  gStyle->SetTitleTextColor(1);
  gStyle->SetTitleFillColor(10);
  gStyle->SetTitleFontSize(0.05);
  // gStyle->SetTitleH(0); // Set the height of the title box
  // gStyle->SetTitleW(0); // Set the width of the title box
  // gStyle->SetTitleX(0); // Set the position of the title box
  // gStyle->SetTitleY(0.985); // Set the position of the title box
  // gStyle->SetTitleStyle(Style_t style = 1001);
  // gStyle->SetTitleBorderSize(2);

  // For the axis titles:

  gStyle->SetTitleColor(1, "XYZ");
  gStyle->SetTitleFont(42, "XYZ");
  gStyle->SetTitleSize(0.06, "XYZ");
  // gStyle->SetTitleXSize(Float_t size = 0.02); // Another way to set the size?
  // gStyle->SetTitleYSize(Float_t size = 0.02);
  gStyle->SetTitleXOffset(0.9);
  gStyle->SetTitleYOffset(1.25);
  // gStyle->SetTitleOffset(1.1, "Y"); // Another way to set the Offset

  // For the axis labels:

  gStyle->SetLabelColor(1, "XYZ");
  gStyle->SetLabelFont(42, "XYZ");
  gStyle->SetLabelOffset(0.007, "XYZ");
  gStyle->SetLabelSize(0.05, "XYZ");

  // For the axis:

  gStyle->SetAxisColor(1, "XYZ");
  gStyle->SetStripDecimals(true);
  gStyle->SetTickLength(0.03, "XYZ");
  gStyle->SetNdivisions(510, "XYZ");
  gStyle->SetPadTickX(1);  // To get tick marks on the opposite side of the frame
  gStyle->SetPadTickY(1);

  // Change for log plots:
  gStyle->SetOptLogx(0);
  gStyle->SetOptLogy(0);
  gStyle->SetOptLogz(0);

  // Postscript options:

  gStyle->SetPaperSize(20.,20.);
  // gStyle->SetLineScalePS(Float_t scale = 3);
  // gStyle->SetLineStyleString(Int_t i, const char* text);
  // gStyle->SetHeaderPS(const char* header);
  // gStyle->SetTitlePS(const char* pstitle);

  // gStyle->SetBarOffset(Float_t baroff = 0.5);
  // gStyle->SetBarWidth(Float_t barwidth = 0.5);
  // gStyle->SetPaintTextFormat(const char* format = "g");
  // gStyle->SetPalette(Int_t ncolors = 0, Int_t* colors = 0);
  // gStyle->SetTimeOffset(Double_t toffset);
  // gStyle->SetHistMinimumZero(true);

  gStyle->SetPalette(1);

  set_plot_style();

  gROOT->ForceStyle();

  TGaxis::SetMaxDigits(4);
  setupcolors();
}


//source: http://ultrahigh.org/2007/08/making-pretty-root-color-palettes/

void set_plot_style()
{
    const Int_t NRGBs = 5;
    const Int_t NCont = 255;

    Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
    Double_t red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
    Double_t green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
    Double_t blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
    TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    gStyle->SetNumberContours(NCont);
}

void setupcolors()
{
    colors.clear();
    styles.clear();
    Color_t array[15] = {1,2,3,4,6,7,8,9,
                         kYellow+3,kOrange+10,kPink-2,kTeal+9,kAzure-8,kViolet-6,kSpring-1};
    for (int i = 0; i < 15; i++)
    {
        colors.push_back(array[i]);
        styles.push_back(1);       //Set the default to 1
                                   //This is to be consistent with the other validation
    }
}
