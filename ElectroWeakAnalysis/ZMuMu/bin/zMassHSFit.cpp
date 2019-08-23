#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "PhysicsTools/Utilities/interface/BreitWigner.h"
#include "PhysicsTools/Utilities/interface/Constant.h"
#include "PhysicsTools/Utilities/interface/Number.h"
#include "PhysicsTools/Utilities/interface/Product.h"
#include "PhysicsTools/Utilities/interface/Sum.h"
#include "PhysicsTools/Utilities/interface/Difference.h"
#include "PhysicsTools/Utilities/interface/Gaussian.h"
#include "PhysicsTools/Utilities/interface/ZLineShape.h"
#include "PhysicsTools/Utilities/interface/Exponential.h"
#include "PhysicsTools/Utilities/interface/Convolution.h"
#include "PhysicsTools/Utilities/interface/HistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/RootMinuit.h"
#include "PhysicsTools/Utilities/interface/RootFunctionAdapter.h"
#include "PhysicsTools/Utilities/interface/rootPlot.h"
#include "TFile.h"
#include "TH1.h"
#include "TF1.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TROOT.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <algorithm>
#include <exception>
#include <iterator>
#include <string>
#include <vector>
using namespace boost;
namespace po = boost::program_options;
using namespace std;

typedef funct::GaussIntegrator IntegratorConv;

// A helper function to simplify the main part.
template <class T>
ostream &operator<<(ostream &os, const vector<T> &v) {
  copy(v.begin(), v.end(), ostream_iterator<T>(cout, " "));
  return os;
}

int main(int ac, char *av[]) {
  try {
    double fMin, fMax;
    string ext;
    po::options_description desc("Allowed options");
    desc.add_options()("help", "produce help message")("include-path,I", po::value<vector<string> >(), "include path")(
        "input-file", po::value<vector<string> >(), "input file")(
        "min,m", po::value<double>(&fMin)->default_value(60), "minimum value for fit range")(
        "max,M", po::value<double>(&fMax)->default_value(120), "maximum value for fit range")(
        "convbwintgamg",
        "fit to the convolution of a breit-wigner plus interference term and gamma propagator and a gaussian")(
        "convexpbwintgamg",
        "fit to the convolution of the product between an exponential and a breit-wigner plus interference term and "
        "gamma propagator with a gaussian")("convbwintgam2gf",
                                            "fit to the convolution of the breit-wigner plus interference term and "
                                            "gamma propagator with a linear combination of fixed gaussians")(
        "output-file,O", po::value<string>(&ext)->default_value(".ps"), "output file format");

    po::positional_options_description p;
    p.add("input-file", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
      cout << "Usage: options_description [options]\n";
      cout << desc;
      return 0;
    }

    if (vm.count("include-path")) {
      cout << "Include paths are: " << vm["include-path"].as<vector<string> >() << "\n";
    }

    vector<string> v_file;
    vector<TH1D *> v_ZMassHistos;
    vector<string> v_eps;

    if (vm.count("input-file")) {
      cout << "Input files are: " << vm["input-file"].as<vector<string> >() << "\n";
      v_file = vm["input-file"].as<vector<string> >();
      for (vector<string>::const_iterator it = v_file.begin(); it != v_file.end(); ++it) {
        TFile *root_file = new TFile(it->c_str(), "read");
        TDirectory *Histos = (TDirectory *)root_file->GetDirectory("ZHisto");
        TDirectory *RecoHistos = (TDirectory *)Histos->GetDirectory("ZRecoHisto");
        TH1D *zMass = (TH1D *)RecoHistos->Get("ZMass");
        zMass->Rebin(4);  //remember...
        zMass->GetXaxis()->SetTitle("#mu #mu invariant mass (GeV/c^{2})");
        v_ZMassHistos.push_back(zMass);
        gROOT->SetStyle("Plain");
        //gStyle->SetOptFit(1111);
        string f_string = *it;
        replace(f_string.begin(), f_string.end(), '.', '_');
        string eps_string = f_string + ext;
        v_eps.push_back(eps_string);
        cout << ">>> histogram loaded\n";
      }
      cout << v_file.size() << ", " << v_ZMassHistos.size() << ", " << v_eps.size() << endl;
      cout << ">>> Input files loaded\n";
    }

    IntegratorConv integratorConv(1.e-5);
    //Values for Z mass and width
    funct::Parameter mass("Mass", 91.364);
    funct::Parameter gamma("Gamma", 4.11);
    //Parameters for Z Line Shape
    funct::Parameter f_gamma("Photon factor", 0.838);
    funct::Parameter f_int("Interference factor", -0.00197);
    //Parameters for fits with gaussians
    funct::Parameter yield("Yield", 283000);
    funct::Parameter alpha("Alpha", 0.771);  //the first gaussian is narrow
    funct::Parameter mean("Mean", 0);        //0.229
    funct::Parameter sigma1("Sigma 1", 0.76);
    funct::Parameter sigma2("Sigma 2", 2.94);
    //Parameter for exponential
    funct::Parameter lambda("Lambda", 0);

    if (vm.count("convbwintgamg")) {
      cout << "Fitting histograms in input files to the convolution of the Breit-Wigner plus Z/photon interference and "
              "photon propagator with a Gaussian\n";
      cout << ">>> set pars: " << endl;
      cout << yield << endl;
      cout << mass << endl;
      cout << gamma << endl;
      cout << f_gamma << endl;
      cout << f_int << endl;
      cout << mean << endl;
      cout << sigma1 << endl;
      for (unsigned int i = 0; i < v_ZMassHistos.size(); ++i) {
        TH1D *zMass = v_ZMassHistos[i];
        funct::ZLineShape zls(mass, gamma, f_gamma, f_int);
        funct::Gaussian gauss(mean, sigma1);
        double range = 3 * sigma1.value();
        funct::Convolution<funct::ZLineShape, funct::Gaussian, IntegratorConv>::type czg(
            zls, gauss, -range, range, integratorConv);
        funct::Constant c(yield);
        typedef funct::Product<funct::Constant,
                               funct::Convolution<funct::ZLineShape, funct::Gaussian, IntegratorConv>::type>::type
            FitFunction;
        FitFunction f = c * czg;
        cout << "set functions" << endl;
        typedef fit::HistoChiSquare<FitFunction> ChiSquared;
        ChiSquared chi2(f, zMass, fMin, fMax);
        int fullBins = chi2.numberOfBins();
        cout << "N. deg. of freedom: " << fullBins << endl;
        fit::RootMinuit<ChiSquared> minuit(chi2, true);
        minuit.addParameter(yield, 10, 100, 10000000);
        minuit.addParameter(mass, .1, 70., 110);
        minuit.addParameter(gamma, 1, 1, 10);
        minuit.addParameter(f_gamma, 0.1, -100, 1000);
        minuit.addParameter(f_int, .0001, -1000000, 1000000);
        minuit.addParameter(mean, 0.001, -0.5, 0.5);
        minuit.fixParameter(mean.name());
        minuit.addParameter(sigma1, 0.1, -5., 5.);
        minuit.minimize();
        minuit.printFitResults();
        vector<shared_ptr<double> > pars;
        pars.push_back(yield.ptr());
        pars.push_back(mass.ptr());
        pars.push_back(gamma.ptr());
        pars.push_back(f_gamma.ptr());
        pars.push_back(f_int.ptr());
        pars.push_back(mean.ptr());
        pars.push_back(sigma1.ptr());
        TF1 fun = root::tf1("fun", f, fMin, fMax, pars);
        fun.SetParNames(yield.name().c_str(),
                        mass.name().c_str(),
                        gamma.name().c_str(),
                        f_gamma.name().c_str(),
                        f_int.name().c_str(),
                        mean.name().c_str(),
                        sigma1.name().c_str());
        fun.SetLineColor(kRed);
        fun.SetNpx(100000);
        TCanvas *canvas = new TCanvas("canvas");
        zMass->Draw("e");
        fun.Draw("same");
        string epsFilename = "ZMassFitCoBwInGaG_" + v_eps[i];
        canvas->SaveAs(epsFilename.c_str());
        canvas->SetLogy();
        string epsLogFilename = "ZMassFitCoBwInGaG_Log_" + v_eps[i];
        canvas->SaveAs(epsLogFilename.c_str());
      }
    }

    if (vm.count("convexpbwintgamg")) {
      cout << "Fitting histograms in input files to the convolution of the product between an exponential and a "
              "breit-wigner plus interference term and gamma propagator with a gaussian"
           << endl;
      cout << ">>> set pars: " << endl;
      cout << yield << endl;
      cout << lambda << endl;
      cout << mass << endl;
      cout << gamma << endl;
      cout << f_gamma << endl;
      cout << f_int << endl;
      cout << mean << endl;
      cout << sigma1 << endl;
      for (unsigned int i = 0; i < v_ZMassHistos.size(); ++i) {
        TH1D *zMass = v_ZMassHistos[i];
        funct::Exponential expo(lambda);
        funct::ZLineShape zls(mass, gamma, f_gamma, f_int);
        funct::Gaussian gauss(mean, sigma1);
        typedef funct::Product<funct::Exponential, funct::ZLineShape>::type ExpZLS;
        ExpZLS expz = expo * zls;
        double range = 3 * sigma1.value();
        funct::Convolution<ExpZLS, funct::Gaussian, IntegratorConv>::type cezg(
            expz, gauss, -range, range, integratorConv);
        funct::Constant c(yield);
        typedef funct::Product<funct::Constant, funct::Convolution<ExpZLS, funct::Gaussian, IntegratorConv>::type>::type
            FitFunction;
        FitFunction f = c * cezg;
        cout << "set functions" << endl;
        typedef fit::HistoChiSquare<FitFunction> ChiSquared;
        ChiSquared chi2(f, zMass, fMin, fMax);
        int fullBins = chi2.numberOfBins();
        cout << "N. deg. of freedom: " << fullBins << endl;
        fit::RootMinuit<ChiSquared> minuit(chi2, true);
        minuit.addParameter(yield, 10, 100, 10000000);
        minuit.addParameter(lambda, 0.1, -100, 100);
        minuit.fixParameter(lambda.name());
        minuit.addParameter(mass, .1, 70., 110);
        minuit.addParameter(gamma, 1, 1, 10);
        minuit.addParameter(f_gamma, 0.1, -100, 1000);
        minuit.addParameter(f_int, .0001, -1000000, 1000000);
        minuit.addParameter(mean, 0.001, -0.5, 0.5);
        minuit.fixParameter(mean.name());
        minuit.addParameter(sigma1, 0.1, -5., 5.);
        minuit.minimize();
        minuit.printFitResults();
        minuit.releaseParameter(lambda.name());
        minuit.minimize();
        minuit.printFitResults();
        vector<shared_ptr<double> > pars;
        pars.push_back(yield.ptr());
        pars.push_back(lambda.ptr());
        pars.push_back(mass.ptr());
        pars.push_back(gamma.ptr());
        pars.push_back(f_gamma.ptr());
        pars.push_back(f_int.ptr());
        pars.push_back(mean.ptr());
        pars.push_back(sigma1.ptr());
        TF1 fun = root::tf1("fun", f, fMin, fMax, pars);
        fun.SetParNames(yield.name().c_str(),
                        lambda.name().c_str(),
                        mass.name().c_str(),
                        gamma.name().c_str(),
                        f_gamma.name().c_str(),
                        f_int.name().c_str(),
                        mean.name().c_str(),
                        sigma1.name().c_str());
        fun.SetLineColor(kRed);
        fun.SetNpx(100000);
        TCanvas *canvas = new TCanvas("canvas");
        zMass->Draw("e");
        fun.Draw("same");
        string epsFilename = "ZMassFitCoExBwInGaG_" + v_eps[i];
        canvas->SaveAs(epsFilename.c_str());
        canvas->SetLogy();
        string epsLogFilename = "ZMassFitCoExBwInGaG_Log_" + v_eps[i];
        canvas->SaveAs(epsLogFilename.c_str());
      }
    }

    if (vm.count("convbwintgam2gf")) {
      cout << "Fitting histograms in input files to the convolution of the Breit-Wigner plus Z/photon interference and "
              "photon propagator with a linear combination of fixed Gaussians\n";
      cout << ">>> set pars: " << endl;
      cout << yield << endl;
      cout << alpha << endl;
      cout << mass << endl;
      cout << gamma << endl;
      cout << f_gamma << endl;
      cout << f_int << endl;
      cout << mean << endl;
      cout << sigma1 << endl;
      cout << sigma2 << endl;
      for (unsigned int i = 0; i < v_ZMassHistos.size(); ++i) {
        TH1D *zMass = v_ZMassHistos[i];
        funct::ZLineShape zls(mass, gamma, f_gamma, f_int);
        funct::Gaussian gaus1(mean, sigma1);
        funct::Gaussian gaus2(mean, sigma2);
        funct::Number _1(1);
        typedef funct::Product<funct::Constant, funct::Gaussian>::type G1;
        typedef funct::Product<funct::Difference<funct::Number, funct::Constant>::type, funct::Gaussian>::type G2;
        typedef funct::Product<funct::Constant, funct::Sum<G1, G2>::type>::type GaussComb;
        funct::Constant c_alpha(alpha), c_yield(yield);
        GaussComb gc = c_yield * (c_alpha * gaus1 + (_1 - c_alpha) * gaus2);
        typedef funct::Convolution<funct::ZLineShape, GaussComb, IntegratorConv>::type FitFunction;
        double range = 3 * max(sigma1.value(), sigma2.value());
        FitFunction f(zls, gc, -range, range, integratorConv);
        cout << "set functions" << endl;
        typedef fit::HistoChiSquare<FitFunction> ChiSquared;
        ChiSquared chi2(f, zMass, fMin, fMax);
        int fullBins = chi2.numberOfBins();
        cout << "N. deg. of freedom: " << fullBins << endl;
        fit::RootMinuit<ChiSquared> minuit(chi2, true);
        minuit.addParameter(yield, 10, 100, 10000000);
        minuit.addParameter(alpha, 0.1, -1., 1.);
        minuit.addParameter(mass, .1, 70., 110);
        minuit.addParameter(gamma, 1, 1, 10);
        minuit.addParameter(f_gamma, 0.1, -100, 1000);
        minuit.addParameter(f_int, .0001, -1000000, 1000000);
        minuit.addParameter(mean, 0.001, -0.5, 0.5);
        minuit.fixParameter(mean.name());
        minuit.addParameter(sigma1, 0.1, -5., 5.);
        minuit.addParameter(sigma2, 0.1, -5., 5.);
        minuit.minimize();
        minuit.printFitResults();
        vector<shared_ptr<double> > pars;
        pars.push_back(yield.ptr());
        pars.push_back(alpha.ptr());
        pars.push_back(mass.ptr());
        pars.push_back(gamma.ptr());
        pars.push_back(f_gamma.ptr());
        pars.push_back(f_int.ptr());
        pars.push_back(mean.ptr());
        pars.push_back(sigma1.ptr());
        pars.push_back(sigma2.ptr());
        TF1 fun = root::tf1("fun", f, fMin, fMax, pars);
        fun.SetParNames(yield.name().c_str(),
                        alpha.name().c_str(),
                        mass.name().c_str(),
                        gamma.name().c_str(),
                        f_gamma.name().c_str(),
                        f_int.name().c_str(),
                        mean.name().c_str(),
                        sigma1.name().c_str(),
                        sigma2.name().c_str());
        fun.SetLineColor(kRed);
        TCanvas *canvas = new TCanvas("canvas");
        zMass->Draw("e");
        fun.Draw("same");
        string epsFilename = "ZMassFitCoBwInGaGGf_" + v_eps[i];
        canvas->SaveAs(epsFilename.c_str());
        canvas->SetLogy();
        string epsLogFilename = "ZMassFitCoBwInGaGGf_Log_" + v_eps[i];
        canvas->SaveAs(epsLogFilename.c_str());
      }
    }

    cout << "It works!\n";
  } catch (std::exception &e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
  } catch (...) {
    cerr << "Exception of unknown type!\n";
  }
  return 0;
}
