#include "../interface/MultiDimFit.h"
#include <stdexcept>
#include <cmath>

#include "TMath.h"
#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooRandom.h"
#include "RooAbsData.h"
#include "RooFitResult.h"
#include "../interface/RooMinimizerOpt.h"
#include <RooStats/ModelConfig.h>
#include "../interface/Combine.h"
#include "../interface/CascadeMinimizer.h"
#include "../interface/CloseCoutSentry.h"
#include "../interface/utils.h"

#include <Math/Minimizer.h>
#include <Math/MinimizerOptions.h>
#include <Math/QuantFuncMathCore.h>
#include <Math/ProbFunc.h>

using namespace RooStats;

MultiDimFit::Algo MultiDimFit::algo_ = None;
MultiDimFit::GridType MultiDimFit::gridType_ = G1x1;
std::vector<std::string>  MultiDimFit::poi_;
std::vector<RooRealVar *> MultiDimFit::poiVars_;
std::vector<float>        MultiDimFit::poiVals_;
RooArgList                MultiDimFit::poiList_;
float                     MultiDimFit::deltaNLL_ = 0;
unsigned int MultiDimFit::points_ = 50;
unsigned int MultiDimFit::firstPoint_ = 0;
unsigned int MultiDimFit::lastPoint_  = std::numeric_limits<unsigned int>::max();
bool MultiDimFit::floatOtherPOIs_ = false;
unsigned int MultiDimFit::nOtherFloatingPoi_ = 0;
bool MultiDimFit::fastScan_ = false;
bool MultiDimFit::hasMaxDeltaNLLForProf_ = false;
float MultiDimFit::maxDeltaNLLForProf_ = 200;


MultiDimFit::MultiDimFit() :
    FitterAlgoBase("MultiDimFit specific options")
{
    options_.add_options()
        ("algo",  boost::program_options::value<std::string>()->default_value("none"), "Algorithm to compute uncertainties")
        ("poi,P",   boost::program_options::value<std::vector<std::string> >(&poi_), "Parameters of interest to fit (default = all)")
        ("floatOtherPOIs",   boost::program_options::value<bool>(&floatOtherPOIs_)->default_value(floatOtherPOIs_), "POIs other than the selected ones will be kept freely floating (1) or fixed (0, default)")
        ("points",  boost::program_options::value<unsigned int>(&points_)->default_value(points_), "Points to use for grid or contour scans")
        ("firstPoint",  boost::program_options::value<unsigned int>(&firstPoint_)->default_value(firstPoint_), "First point to use")
        ("lastPoint",  boost::program_options::value<unsigned int>(&lastPoint_)->default_value(lastPoint_), "Last point to use")
        ("fastScan", "Do a fast scan, evaluating the likelihood without profiling it.")
        ("maxDeltaNLLForProf",  boost::program_options::value<float>(&maxDeltaNLLForProf_)->default_value(maxDeltaNLLForProf_), "Last point to use")
       ;
}

void MultiDimFit::applyOptions(const boost::program_options::variables_map &vm) 
{
    applyOptionsBase(vm);
    std::string algo = vm["algo"].as<std::string>();
    if (algo == "none") {
        algo_ = None;
    } else if (algo == "singles") {
        algo_ = Singles;
    } else if (algo == "cross") {
        algo_ = Cross;
    } else if (algo == "grid" || algo == "grid3x3" ) {
        algo_ = Grid; gridType_ = G1x1;
        if (algo == "grid3x3") gridType_ = G3x3;
    } else if (algo == "random") {
        algo_ = RandomPoints;
    } else if (algo == "contour2d") {
        algo_ = Contour2D;
    } else throw std::invalid_argument(std::string("Unknown algorithm: "+algo));
    fastScan_ = (vm.count("fastScan") > 0);
    hasMaxDeltaNLLForProf_ = !vm["maxDeltaNLLForProf"].defaulted();
}

bool MultiDimFit::runSpecific(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) { 
    // one-time initialization of POI variables, TTree branches, ...
    static int isInit = false;
    if (!isInit) { initOnce(w, mc_s); isInit = true; }

    // Get PDF
    RooAbsPdf &pdf = *mc_s->GetPdf();

    // Process POI not in list
    nOtherFloatingPoi_ = 0;
    RooLinkedListIter iterP = mc_s->GetParametersOfInterest()->iterator();
    for (RooAbsArg *a = (RooAbsArg*) iterP.Next(); a != 0; a = (RooAbsArg*) iterP.Next()) {
        if (poiList_.contains(*a)) continue;
        RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
        if (rrv == 0) { std::cerr << "MultiDimFit: Parameter of interest " << a->GetName() << " which is not a RooRealVar will be ignored" << std::endl; continue; }
        rrv->setConstant(!floatOtherPOIs_);
        if (floatOtherPOIs_) nOtherFloatingPoi_++;
    }
 
    // start with a best fit
    const RooCmdArg &constrainCmdArg = withSystematics  ? RooFit::Constrain(*mc_s->GetNuisanceParameters()) : RooFit::NumCPU(1);
    std::auto_ptr<RooFitResult> res(doFit(pdf, data, (algo_ == Singles ? poiList_ : RooArgList()), constrainCmdArg, false, 1, true)); 
    if (res.get() || keepFailures_) {
        for (int i = 0, n = poi_.size(); i < n; ++i) {
            poiVals_[i] = poiVars_[i]->getVal();
        }
        if (algo_ != None) Combine::commitPoint(/*expected=*/false, /*quantile=*/1.); // otherwise we get it multiple times
    }
    std::auto_ptr<RooAbsReal> nll;
    if (algo_ != None && algo_ != Singles) {
        nll.reset(pdf.createNLL(data, constrainCmdArg, RooFit::Extended(pdf.canBeExtended())));
    } 
    switch(algo_) {
        case None: 
            if (verbose > 0) {
                std::cout << "\n --- MultiDimFit ---" << std::endl;
                std::cout << "best fit parameter values: "  << std::endl;
                int len = poi_[0].length();
                for (int i = 0, n = poi_.size(); i < n; ++i) {
                    len = std::max<int>(len, poi_[i].length());
                }
                for (int i = 0, n = poi_.size(); i < n; ++i) {
                    printf("   %*s :  %+8.3f\n", len, poi_[i].c_str(), poiVals_[i]);
                }
            }
            break;
        case Singles: if (res.get()) doSingles(*res); break;
        case Cross: doBox(*nll, cl, "box", true); break;
        case Grid: doGrid(*nll); break;
        case RandomPoints: doRandomPoints(*nll); break;
        case Contour2D: doContour2D(*nll); break;
    }
    
    return true;
}

void MultiDimFit::initOnce(RooWorkspace *w, RooStats::ModelConfig *mc_s) {
    RooArgSet mcPoi(*mc_s->GetParametersOfInterest());
    if (poi_.empty()) {
        RooLinkedListIter iterP = mc_s->GetParametersOfInterest()->iterator();
        for (RooAbsArg *a = (RooAbsArg*) iterP.Next(); a != 0; a = (RooAbsArg*) iterP.Next()) {
            poi_.push_back(a->GetName());
        }
    }
    for (std::vector<std::string>::const_iterator it = poi_.begin(), ed = poi_.end(); it != ed; ++it) {
        RooAbsArg *a = mcPoi.find(it->c_str());
        if (a == 0) throw std::invalid_argument(std::string("Parameter of interest ")+*it+" not in model.");
        RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
        if (rrv == 0) throw std::invalid_argument(std::string("Parameter of interest ")+*it+" not a RooRealVar.");
        poiVars_.push_back(rrv);
        poiVals_.push_back(rrv->getVal());
        poiList_.add(*rrv);
    }
    // then add the branches to the tree (at the end, so there are no resizes)
    for (int i = 0, n = poi_.size(); i < n; ++i) {
        Combine::addBranch(poi_[i].c_str(), &poiVals_[i], (poi_[i]+"/F").c_str()); 
    }
    Combine::addBranch("deltaNLL", &deltaNLL_, "deltaNLL/F");
}

void MultiDimFit::doSingles(RooFitResult &res)
{
    std::cout << "\n --- MultiDimFit ---" << std::endl;
    std::cout << "best fit parameter values and profile-likelihood uncertainties: "  << std::endl;
    int len = poi_[0].length();
    for (int i = 0, n = poi_.size(); i < n; ++i) {
        len = std::max<int>(len, poi_[i].length());
    }
    for (int i = 0, n = poi_.size(); i < n; ++i) {
        RooRealVar *rf = dynamic_cast<RooRealVar*>(res.floatParsFinal().find(poi_[i].c_str()));
        double bestFitVal = rf->getVal();

        double hiErr = +(rf->hasRange("err68") ? rf->getMax("err68") - bestFitVal : rf->getAsymErrorHi());
        double loErr = -(rf->hasRange("err68") ? rf->getMin("err68") - bestFitVal : rf->getAsymErrorLo());
        double maxError = std::max<double>(std::max<double>(hiErr, loErr), rf->getError());

        if (fabs(hiErr) < 0.001*maxError) hiErr = -bestFitVal + rf->getMax();
        if (fabs(loErr) < 0.001*maxError) loErr = +bestFitVal - rf->getMin();

        double hiErr95 = +(do95_ && rf->hasRange("err95") ? rf->getMax("err95") - bestFitVal : 0);
        double loErr95 = -(do95_ && rf->hasRange("err95") ? rf->getMin("err95") - bestFitVal : 0);

        poiVals_[i] = bestFitVal - loErr; Combine::commitPoint(true, /*quantile=*/0.32);
        poiVals_[i] = bestFitVal + hiErr; Combine::commitPoint(true, /*quantile=*/0.32);
        if (do95_ && rf->hasRange("err95")) {
            poiVals_[i] = rf->getMax("err95"); Combine::commitPoint(true, /*quantile=*/0.05);
            poiVals_[i] = rf->getMin("err95"); Combine::commitPoint(true, /*quantile=*/0.05);
            poiVals_[i] = bestFitVal;
            printf("   %*s :  %+8.3f   %+6.3f/%+6.3f (68%%)    %+6.3f/%+6.3f (95%%) \n", len, poi_[i].c_str(), 
                    poiVals_[i], -loErr, hiErr, loErr95, -hiErr95);
        } else {
            poiVals_[i] = bestFitVal;
            printf("   %*s :  %+8.3f   %+6.3f/%+6.3f (68%%)\n", len, poi_[i].c_str(), 
                    poiVals_[i], -loErr, hiErr);
        }
    }
}

void MultiDimFit::doGrid(RooAbsReal &nll) 
{
    unsigned int n = poi_.size();
    if (poi_.size() > 2) throw std::logic_error("Don't know how to do a grid with more than 2 POIs.");
    double nll0 = nll.getVal();

    std::vector<double> p0(n), pmin(n), pmax(n);
    for (unsigned int i = 0; i < n; ++i) {
        p0[i] = poiVars_[i]->getVal();
        pmin[i] = poiVars_[i]->getMin();
        pmax[i] = poiVars_[i]->getMax();
        poiVars_[i]->setConstant(true);
    }

    CascadeMinimizer minim(nll, CascadeMinimizer::Constrained);
    minim.setStrategy(minimizerStrategy_);
    std::auto_ptr<RooArgSet> params(nll.getParameters((const RooArgSet *)0));
    RooArgSet snap; params->snapshot(snap);
    //snap.Print("V");
    if (n == 1) {
        for (unsigned int i = 0; i < points_; ++i) {
            if (i < firstPoint_) continue;
            if (i > lastPoint_)  break;
            double x =  pmin[0] + (i+0.5)*(pmax[0]-pmin[0])/points_; 
            if (verbose > 1) std::cout << "Point " << i << "/" << points_ << " " << poiVars_[0]->GetName() << " = " << x << std::endl;
            *params = snap; 
            poiVals_[0] = x;
            poiVars_[0]->setVal(x);
            // now we minimize
            bool ok = fastScan_ || (hasMaxDeltaNLLForProf_ && (nll.getVal() - nll0) > maxDeltaNLLForProf_) ? 
                        true : 
                        minim.minimize(verbose-1);
            if (ok) {
                deltaNLL_ = nll.getVal() - nll0;
                double qN = 2*(deltaNLL_);
                double prob = ROOT::Math::chisquared_cdf_c(qN, n+nOtherFloatingPoi_);
                Combine::commitPoint(true, /*quantile=*/prob);
            }
        }
    } else if (n == 2) {
        unsigned int sqrn = ceil(sqrt(double(points_)));
        unsigned int ipoint = 0, nprint = ceil(0.005*sqrn*sqrn);
        RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CountErrors);
        CloseCoutSentry sentry(verbose < 2);
        double deltaX =  (pmax[0]-pmin[0])/sqrn, deltaY = (pmax[1]-pmin[1])/sqrn;
        for (unsigned int i = 0; i < sqrn; ++i) {
            for (unsigned int j = 0; j < sqrn; ++j, ++ipoint) {
                if (ipoint < firstPoint_) continue;
                if (ipoint > lastPoint_)  break;
                *params = snap; 
                double x =  pmin[0] + (i+0.5)*deltaX; 
                double y =  pmin[1] + (j+0.5)*deltaY; 
                if (verbose && (ipoint % nprint == 0)) {
                         fprintf(sentry.trueStdOut(), "Point %d/%d, (i,j) = (%d,%d), %s = %f, %s = %f\n",
                                        ipoint,sqrn*sqrn, i,j, poiVars_[0]->GetName(), x, poiVars_[1]->GetName(), y);
                }
                poiVals_[0] = x;
                poiVals_[1] = y;
                poiVars_[0]->setVal(x);
                poiVars_[1]->setVal(y);
                nll.clearEvalErrorLog(); nll.getVal();
                if (nll.numEvalErrors() > 0) { 
                    deltaNLL_ = 9999; Combine::commitPoint(true, /*quantile=*/0); 
                    continue;
                }
                // now we minimize
                bool skipme = hasMaxDeltaNLLForProf_ && (nll.getVal() - nll0) > maxDeltaNLLForProf_;
                bool ok = fastScan_ || skipme ? true :  minim.minimize(verbose-1);
                if (ok) {
                    deltaNLL_ = nll.getVal() - nll0;
                    double qN = 2*(deltaNLL_);
                    double prob = ROOT::Math::chisquared_cdf_c(qN, n+nOtherFloatingPoi_);
                    Combine::commitPoint(true, /*quantile=*/prob);
                }
                if (gridType_ == G3x3 && !skipme) {
                    double x0 = x, y0 = y;
                    for (int i2 = -1; i2 <= +1; ++i2) {
                        for (int j2 = -1; j2 <= +1; ++j2) {
                            if (i2 == 0 && j2 == 0) continue;
                            x = x0 + 0.33333333*i2*deltaX;
                            y = y0 + 0.33333333*j2*deltaY;
                            poiVals_[0] = x; poiVars_[0]->setVal(x);
                            poiVals_[1] = y; poiVars_[1]->setVal(y);
                            nll.clearEvalErrorLog(); nll.getVal();
                            if (nll.numEvalErrors() > 0) { 
                                deltaNLL_ = 9999; Combine::commitPoint(true, /*quantile=*/0); 
                                continue;
                            }
                            deltaNLL_ = nll.getVal() - nll0;
                            if (!fastScan_ && std::min(fabs(deltaNLL_ - 1.15), fabs(deltaNLL_ - 2.995)) < 0.3) {
                                minim.minimize(verbose-1);
                                deltaNLL_ = nll.getVal() - nll0;
                            }
                            double qN = 2*(deltaNLL_);
                            double prob = ROOT::Math::chisquared_cdf_c(qN, n+nOtherFloatingPoi_);
                            Combine::commitPoint(true, /*quantile=*/prob);
                        }
                    }
                }
            }
        }
    }
}

void MultiDimFit::doRandomPoints(RooAbsReal &nll) 
{
    double nll0 = nll.getVal();
    for (unsigned int i = 0, n = poi_.size(); i < n; ++i) {
        poiVars_[i]->setConstant(true);
    }

    CascadeMinimizer minim(nll, CascadeMinimizer::Constrained);
    minim.setStrategy(minimizerStrategy_);
    unsigned int n = poi_.size();
    for (unsigned int j = 0; j < points_; ++j) {
        for (unsigned int i = 0; i < n; ++i) {
            poiVars_[i]->randomize();
            poiVals_[i] = poiVars_[i]->getVal(); 
        }
        // now we minimize
        {   
            CloseCoutSentry sentry(verbose < 3);    
            bool ok = minim.minimize(verbose-1);
            if (ok) {
                double qN = 2*(nll.getVal() - nll0);
                double prob = ROOT::Math::chisquared_cdf_c(qN, n+nOtherFloatingPoi_);
                Combine::commitPoint(true, /*quantile=*/prob);
            }
        } 
    }
}

void MultiDimFit::doContour2D(RooAbsReal &nll) 
{
    if (poi_.size() != 2) throw std::logic_error("Contour2D works only in 2 dimensions");
    RooRealVar *xv = poiVars_[0]; double x0 = poiVals_[0]; float &x = poiVals_[0];
    RooRealVar *yv = poiVars_[1]; double y0 = poiVals_[1]; float &y = poiVals_[1];

    double threshold = nll.getVal() + 0.5*ROOT::Math::chisquared_quantile_c(1-cl,2+nOtherFloatingPoi_);
    if (verbose>0) std::cout << "Best fit point is for " << xv->GetName() << ", "  << yv->GetName() << " =  " << x0 << ", " << y0 << std::endl;

    // make a box
    doBox(nll, cl, "box");
    double xMin = xv->getMin("box"), xMax = xv->getMax("box");
    double yMin = yv->getMin("box"), yMax = yv->getMax("box");

    verbose--; // reduce verbosity to avoid messages from findCrossing
    // ===== Get relative min/max of x for several fixed y values =====
    yv->setConstant(true);
    for (unsigned int j = 0; j <= points_; ++j) {
        if (j < firstPoint_) continue;
        if (j > lastPoint_)  break;
        // take points uniformly spaced in polar angle in the case of a perfect circle
        double yc = 0.5*(yMax + yMin), yr = 0.5*(yMax - yMin);
        yv->setVal( yc + yr * std::cos(j*M_PI/double(points_)) );
        // ===== Get the best fit x (could also do without profiling??) =====
        xv->setConstant(false);  xv->setVal(x0);
        CascadeMinimizer minimXI(nll, CascadeMinimizer::Unconstrained, xv);
        minimXI.setStrategy(minimizerStrategy_);
        {
            CloseCoutSentry sentry(verbose < 3);    
            minimXI.minimize(verbose-1);
        }
        double xc = xv->getVal(); xv->setConstant(true);
        if (verbose>-1) std::cout << "Best fit " << xv->GetName() << " for  " << yv->GetName() << " = " << yv->getVal() << " is at " << xc << std::endl;
        // ===== Then get the range =====
        CascadeMinimizer minim(nll, CascadeMinimizer::Constrained);
        double xup = findCrossing(minim, nll, *xv, threshold, xc, xMax);
        if (!std::isnan(xup)) { 
            x = xup; y = yv->getVal(); Combine::commitPoint(true, /*quantile=*/1-cl);
            if (verbose>-1) std::cout << "Minimum of " << xv->GetName() << " at " << cl << " CL for " << yv->GetName() << " = " << y << " is " << x << std::endl;
        }
        
        double xdn = findCrossing(minim, nll, *xv, threshold, xc, xMin);
        if (!std::isnan(xdn)) { 
            x = xdn; y = yv->getVal(); Combine::commitPoint(true, /*quantile=*/1-cl);
            if (verbose>-1) std::cout << "Maximum of " << xv->GetName() << " at " << cl << " CL for " << yv->GetName() << " = " << y << " is " << x << std::endl;
        }
    }

    verbose++; // restore verbosity
}
void MultiDimFit::doBox(RooAbsReal &nll, double cl, const char *name, bool commitPoints)  {
    unsigned int n = poi_.size();
    double nll0 = nll.getVal(), threshold = nll0 + 0.5*ROOT::Math::chisquared_quantile_c(1-cl,n+nOtherFloatingPoi_);

    std::vector<double> p0(n);
    for (unsigned int i = 0; i < n; ++i) {
        p0[i] = poiVars_[i]->getVal();
        poiVars_[i]->setConstant(false);
    }

    verbose--; // reduce verbosity due to findCrossing
    for (unsigned int i = 0; i < n; ++i) {
        RooRealVar *xv = poiVars_[i];
        xv->setConstant(true);
        CascadeMinimizer minimX(nll, CascadeMinimizer::Constrained);
        minimX.setStrategy(minimizerStrategy_);

        for (unsigned int j = 0; j < n; ++j) poiVars_[j]->setVal(p0[j]);
        double xMin = findCrossing(minimX, nll, *xv, threshold, p0[i], xv->getMin()); 
        if (!std::isnan(xMin)) { 
            if (verbose > -1) std::cout << "Minimum of " << xv->GetName() << " at " << cl << " CL for all others floating is " << xMin << std::endl;
            for (unsigned int j = 0; j < n; ++j) poiVals_[j] = poiVars_[j]->getVal();
            if (commitPoints) Combine::commitPoint(true, /*quantile=*/1-cl);
        } else {
            xMin = xv->getMin();
            for (unsigned int j = 0; j < n; ++j) poiVals_[j] = poiVars_[j]->getVal();
            double prob = ROOT::Math::chisquared_cdf_c(2*(nll.getVal() - nll0), n+nOtherFloatingPoi_);
            if (commitPoints) Combine::commitPoint(true, /*quantile=*/prob);
            if (verbose > -1) std::cout << "Minimum of " << xv->GetName() << " at " << cl << " CL for all others floating is " << xMin << " (on the boundary, p-val " << prob << ")" << std::endl;
        }
        
        for (unsigned int j = 0; j < n; ++j) poiVars_[j]->setVal(p0[j]);
        double xMax = findCrossing(minimX, nll, *xv, threshold, p0[i], xv->getMax()); 
        if (!std::isnan(xMax)) { 
            if (verbose > -1) std::cout << "Maximum of " << xv->GetName() << " at " << cl << " CL for all others floating is " << xMax << std::endl;
            for (unsigned int j = 0; j < n; ++j) poiVals_[j] = poiVars_[j]->getVal();
            if (commitPoints) Combine::commitPoint(true, /*quantile=*/1-cl);
        } else {
            xMax = xv->getMax();
            double prob = ROOT::Math::chisquared_cdf_c(2*(nll.getVal() - nll0), n+nOtherFloatingPoi_);
            for (unsigned int j = 0; j < n; ++j) poiVals_[j] = poiVars_[j]->getVal();
            if (commitPoints) Combine::commitPoint(true, /*quantile=*/prob);
            if (verbose > -1) std::cout << "Maximum of " << xv->GetName() << " at " << cl << " CL for all others floating is " << xMax << " (on the boundary, p-val " << prob << ")" << std::endl;
        }

        xv->setRange(name, xMin, xMax);
        xv->setConstant(false);
    }
    verbose++; // restore verbosity 
}
