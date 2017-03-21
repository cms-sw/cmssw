#include <iostream>
#include <cassert>
#include <sstream>
#include <cfloat>
#include <algorithm>
#include <memory>

#include "CondTools/Hcal/interface/visualizeHFPhase1PMTParams.h"
#include "CondTools/Hcal/interface/parseHcalDetId.h"

#include "TCanvas.h"
#include "TPad.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TH1F.h"

typedef std::vector<std::shared_ptr<TGraph> > Garbage;

//
// Return a usable root name (without spaces) for the given detector id
//
static std::string rootDetectorName(const HcalDetId& id)
{
    using namespace std;
    ostringstream os;
    os << hcalSubdetectorName(id.subdet())
       << '_' << id.ieta()
       << '_' << id.iphi()
       << '_' << id.depth();
    return os.str();
}


static TH1F* plotFunctors(TVirtualPad* pad,
                          const char* plotTitle,
                          const char* yAxisTitle,
                          const double cut,
                          const double ymin,
                          const double ymax,
                          const AbsHcalFunctor& f1,
                          const AbsHcalFunctor& f2,
                          const VisualizationOptions& opts,
                          Garbage& garbage,
                          const bool isReference,
                          TH1F* frame)
{
    const int lineWidth = 2;
    const int refLineWidth = 4;

    assert(opts.plotPoints > 1);
    const unsigned nDraw = opts.plotPoints + 1;
    std::vector<double> xvec(nDraw);
    std::vector<double> y1(nDraw);
    std::vector<double> y2(nDraw);
    const double xrange = opts.maxCharge - cut;
    const double yrange = ymax - ymin;
    xvec[0] = cut;
    y1[0] = ymin + 0.05*yrange;
    y2[0] = ymax - 0.05*yrange;
    const double h = xrange/(opts.plotPoints - 1);
    for (unsigned i=1; i<=opts.plotPoints; ++i)
    {
        const double x = cut + h*(i - 1);
        xvec[i] = x;
        y1[i] = f1(x);
        y2[i] = f2(x);
    }

    // Root was designed by (insert your favorite explective here),
    // so the plot frame on the pad is of type TH1F (a histogram!)
    //
    if (frame == nullptr)
    {
        frame = pad->DrawFrame(opts.minCharge, ymin,
                               opts.maxCharge + 0.1*xrange, ymax);
        frame->GetXaxis()->SetTitle("Q [fC]");
        frame->GetYaxis()->SetTitle(yAxisTitle);
        frame->SetTitle(plotTitle);
    }

    std::shared_ptr<TGraph> gr1(new TGraph(nDraw, &xvec[0], &y1[0]));
    gr1->SetLineWidth(isReference ? refLineWidth : lineWidth);
    gr1->SetLineColor(isReference ? kGreen : kBlue);
    gr1->Draw("L");
    garbage.push_back(gr1);

    std::shared_ptr<TGraph> gr2(new TGraph(nDraw, &xvec[0], &y2[0]));
    gr2->SetLineWidth(isReference ? refLineWidth : lineWidth);
    gr2->SetLineColor(isReference ? kGreen : kRed);
    gr2->Draw("L");
    garbage.push_back(gr2);

    return frame;
}


static void plotConfig(const HcalDetId& id,
                       const unsigned tableIndex,
                       const HFPhase1PMTData& cut,
                       const VisualizationOptions& opts,
                       const HFPhase1PMTData* ref)
{
    using namespace std;

    const double canvasWidth = 1000;
    const double canvasHeight = 300;

    // Root was designed by idiots oblivious to smart pointers and unable
    // to carry their objects around. Because of this, most plottable
    // object in "Root" must have a unique name, even if it makes no sense
    // whatsoever to name such an object. Keep it in mind that idiots are
    // incapable of handling spaces inside the names.
    //
    const bool isDefault = tableIndex == HcalIndexLookup::InvalidIndex;
    const std::string& detName = rootDetectorName(id);
    const std::string& canvName = isDefault ? std::string("Default") :
                                              detName;
    ostringstream ostitle;
    if (isDefault)
        ostitle << "Default Cuts";
    else
        ostitle << "Cuts for " << detName;
    const std::string& canvTitle = ostitle.str();

    TCanvas* canv = new TCanvas(canvName.c_str(), canvTitle.c_str(),
                                canvasWidth, canvasHeight);

    // Root was designed by idiots oblivious to the difference between
    // window size and canvas size. Because of this, the canvas size
    // has to be set explicitly again, despite the fact that the
    // constructor already had these arguments.
    //
    canv->SetCanvasSize(canvasWidth, canvasHeight);
    canv->Divide(3, 1);

    // Root was designed by idiots changing their minds too often.
    // Graphs are an exception from the naming scheme and ownership
    // convention. They are not managed by root, so graphs on all
    // pads must persist until the whole canvas is written out.
    // This is why we need a garbage bin -- to manage these things.
    //
    Garbage bin;

    TVirtualPad* pad = canv->cd(1);
    {   
        TH1F* frame = nullptr;
        if (ref)
            frame = plotFunctors(pad, "PMT Anode 0", "Rise Time Cuts [ns]",
                                 ref->minCharge0(), opts.minTDC, opts.maxTDC,
                                 ref->cut(HFPhase1PMTData::T_0_MIN),
                                 ref->cut(HFPhase1PMTData::T_0_MAX),
                                 opts, bin, true, frame);
        plotFunctors(pad, "PMT Anode 0", "Rise Time Cuts [ns]",
                     cut.minCharge0(), opts.minTDC, opts.maxTDC,
                     cut.cut(HFPhase1PMTData::T_0_MIN),
                     cut.cut(HFPhase1PMTData::T_0_MAX),
                     opts, bin, false, frame);
    }
    pad->Draw();

    ostringstream midtitle;
    midtitle << "PMT Anode 1 ";
    if (isDefault)
        midtitle << "(Default Cuts)";
    else
        midtitle << '(' << detName << ", Index " << tableIndex << ')';
    const string& mtstr = midtitle.str();
    pad = canv->cd(2);
    {
        TH1F* frame = nullptr;
        if (ref)
            frame = plotFunctors(pad, mtstr.c_str(), "Rise Time Cuts [ns]",
                                 ref->minCharge1(), opts.minTDC, opts.maxTDC,
                                 ref->cut(HFPhase1PMTData::T_1_MIN),
                                 ref->cut(HFPhase1PMTData::T_1_MAX),
                                 opts, bin, true, frame);
        plotFunctors(pad, mtstr.c_str(), "Rise Time Cuts [ns]",
                     cut.minCharge1(), opts.minTDC, opts.maxTDC,
                     cut.cut(HFPhase1PMTData::T_1_MIN),
                     cut.cut(HFPhase1PMTData::T_1_MAX),
                     opts, bin, false, frame);
    }
    pad->Draw();

    pad = canv->cd(3);
    {
        TH1F* frame = nullptr;
        if (ref)
            frame = plotFunctors(pad, "Charge Asymmetry", "Asymmetry Cuts",
                                 ref->minChargeAsymm(), opts.minAsymm, opts.maxAsymm,
                                 ref->cut(HFPhase1PMTData::ASYMM_MIN),
                                 ref->cut(HFPhase1PMTData::ASYMM_MAX),
                                 opts, bin, true, frame);
        plotFunctors(pad, "Charge Asymmetry", "Asymmetry Cuts",
                     cut.minChargeAsymm(), opts.minAsymm, opts.maxAsymm,
                     cut.cut(HFPhase1PMTData::ASYMM_MIN),
                     cut.cut(HFPhase1PMTData::ASYMM_MAX),
                     opts, bin, false, frame);
    }
    pad->Draw();

    canv->Write();
}


void visualizeHFPhase1PMTParams(const std::vector<HcalDetId>& idVec,
                                const HFPhase1PMTParams& cuts,
                                const VisualizationOptions& options,
                                const HFPhase1PMTParams* reference)
{
    using namespace std;

    const unsigned n = idVec.size();
    if (options.verbose)
    {
        cout << "Cut visualization is requested for the following PMTs:\n";
        for (unsigned i=0; i<n; ++i)
            cout << idVec[i] << '\n';
        cout.flush();
    }

    // Plot the default cut
    const HFPhase1PMTData* defaultCut = cuts.getDefault();
    const HFPhase1PMTData* defaultRef = nullptr;
    if (reference)
        defaultRef = reference->getDefault();
    if (defaultCut)
        plotConfig(HcalDetId(), HcalIndexLookup::InvalidIndex,
                   *defaultCut, options, defaultRef);

    // Plot all other cuts
    for (unsigned i=0; i<n; ++i)
    {
        const HcalDetId& id = idVec[i];
        const unsigned tableIndex = cuts.getIndex(id);
        if (tableIndex == HcalIndexLookup::InvalidIndex)
        {
            if (defaultCut)
                cout << "PMT " << id << " is using default config" << endl;
            else
                cout << "ERROR! No configuration found for PMT " << id << endl;
        }
        else
        {
            cout << "PMT " << id << " is using config " << tableIndex << endl;
            const HFPhase1PMTData* cut = cuts.getByIndex(tableIndex);
            assert(cut);
            const HFPhase1PMTData* ref = nullptr;
            if (reference)
                ref = reference->getByIndex(tableIndex);
            plotConfig(id, tableIndex, *cut, options, ref);
        }
    }
}
