#include "CMGTools/TTHAnalysis/interface/DistributionRemapper.h"

#include <cmath>
#include <cstdio>
#include <TH1.h>
#include <TAxis.h>

DistributionRemapper::DistributionRemapper(const TH1 *source, const TH1 *target) :
    xmin_(source->GetXaxis()->GetXmin()),
    ymin_(target->GetXaxis()->GetXmin()),
    xmax_(source->GetXaxis()->GetXmax()),
    ymax_(target->GetXaxis()->GetXmax()),
    x_(source->GetNbinsX()+1),
    y_(source->GetNbinsX()+1),
    interp_(0)
{
    int ns = source->GetNbinsX();
    int nt = target->GetNbinsX(); 
    const TAxis *axs = source->GetXaxis();
    const TAxis *axt = target->GetXaxis(); 

    std::vector<double> xt, yt;
    xt.resize(nt+3);
    yt.resize(nt+3);

    // make inverse cdf of target
    //printf("preparing dataset for inverse cdf of target (%d bins)\n", nt);
    double tnorm = 1.0/target->Integral(0, nt+1);
    xt[0] = axt->GetXmin() - 0.5*(axt->GetXmax()-axt->GetXmin());
    yt[0] = 0.0;
    int j = 0;
    if (target->GetBinContent(0) != 0) {
        j = 1;
        xt[1] = axt->GetBinLowEdge(1);
        yt[1] = tnorm*target->GetBinContent(0);
    }
    for (int i = 1; i <= nt; ++i) {
        if (target->GetBinContent(i) == 0) continue;
        j++;
        xt[j] = axt->GetBinUpEdge(i);
        yt[j] = yt[j-1] + tnorm*target->GetBinContent(i);
    }
    if (yt[j] < 1) {
        j++;
        yt[j] = 1.0;
        xt[j] = axt->GetXmax() + 0.5*(axt->GetXmax()-axt->GetXmin());
    }
    xt.resize(j+1);
    yt.resize(j+1);

    //printf("creating interpolator\n");
    ROOT::Math::Interpolator tinv(yt, xt, ROOT::Math::Interpolation::kLINEAR);
    
    // loop over bin
    //printf("preparing dataset for morphing function (%d bins)\n", ns);
    double snorm = 1.0/source->Integral(0, ns+1);
    double srun = snorm*source->GetBinContent(0);
    x_[0] = axs->GetBinLowEdge(1);
    y_[0] = tinv.Eval(srun);
    for (int i = 1; i <= ns; ++i) {
        srun += tnorm * source->GetBinContent(i);
        x_[i] = axt->GetBinUpEdge(i);
        y_[i] = tinv.Eval(srun);
    }
}

DistributionRemapper::~DistributionRemapper() 
{
    delete interp_; interp_ = 0;
}

double DistributionRemapper::Eval(double x) const 
{
    if (!interp_) init();
    if (x < xmin_) return ymin_;
    if (x > xmax_) return ymax_;
    return interp_->Eval(x);
}

void DistributionRemapper::init() const 
{
    interp_ = new ROOT::Math::Interpolator(x_, y_, ROOT::Math::Interpolation::kCSPLINE);
}
