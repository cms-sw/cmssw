#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "PhysicsTools/Utilities/interface/ZLineShape.h"
#include "PhysicsTools/Utilities/interface/Gaussian.h"
#include "PhysicsTools/Utilities/interface/Numerical.h"
#include "PhysicsTools/Utilities/interface/Exponential.h"
#include "PhysicsTools/Utilities/interface/Polynomial.h"
#include "PhysicsTools/Utilities/interface/Constant.h"
#include "PhysicsTools/Utilities/interface/Convolution.h"
#include "PhysicsTools/Utilities/interface/Operations.h"
#include "PhysicsTools/Utilities/interface/Integral.h"
#include "PhysicsTools/Utilities/interface/MultiHistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/RootMinuit.h"
#include "PhysicsTools/Utilities/interface/RootMinuitCommands.h"
#include "PhysicsTools/Utilities/interface/FunctClone.h"
#include "PhysicsTools/Utilities/interface/rootPlot.h"
#include "TROOT.h"
#include "TH1.h"
#include "TFile.h"
#include <boost/program_options.hpp>
using namespace boost;
namespace po = boost::program_options;

#include <iostream>
#include <algorithm> 
#include <exception>
#include <iterator>
#include <string>
#include <vector>
using namespace std;

// A helper function to simplify the main part.
template<class T>
ostream& operator<<(ostream& os, const vector<T>& v) {
  copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
  return os;
}

//A function that sets istogram contents to 0 
//if they are too small
void fix(TH1* histo) {
  for(int i = 1; i <= histo->GetNbinsX(); ++i) {
    if(histo->GetBinContent(i) < 0.1) {
      histo->SetBinContent(i, 0.0);
      histo->SetBinError(i, 0.0);
    }
  }
}

typedef funct::GaussIntegrator IntegratorConv;
typedef funct::GaussIntegrator IntegratorNorm;

typedef funct::Product<funct::Exponential, 
		       funct::Convolution<funct::ZLineShape, funct::Gaussian, IntegratorConv>::type>::type ZPeak;

typedef funct::Master<ZPeak> SigPeak;
typedef funct::Slave<ZPeak> SigPeakClone;
typedef funct::Product<funct::Parameter, SigPeak>::type Sig1;
typedef funct::Product<funct::Parameter, SigPeakClone>::type Sig2;

typedef funct::Product<
            funct::Exponential, 
            funct::Polynomial<2> >::type ExpPoly;

NUMERICAL_FUNCT_INTEGRAL(ExpPoly, GaussIntegrator);

typedef funct::DefIntegral<ExpPoly, funct::Constant, funct::Constant, IntegratorNorm> ExpPolyNormFactor;
typedef funct::Ratio<ExpPoly, ExpPolyNormFactor>::type ExpPolyNorm;

typedef funct::Product<
          funct::Parameter, 
          funct::Difference<
            funct::Ratio<
              funct::Numerical<2>, 
              funct::Parameter>::type,
            funct::Numerical<2> >::type>::type Coeff1;

typedef funct::Product<Coeff1, ExpPolyNorm>::type Bkg1;

typedef funct::Product<
          funct::Parameter, 
          funct::Square<
            funct::Difference<
              funct::Ratio<
                funct::Numerical<1>, 
                funct::Parameter>::type,
              funct::Numerical<1> >::type>::type>::type Coeff2;

typedef funct::Product<Coeff2, ExpPolyNorm>::type Bkg2;

NUMERICAL_FUNCT_INTEGRAL(Bkg1, GaussIntegrator);
NUMERICAL_FUNCT_INTEGRAL(Bkg2, GaussIntegrator);

typedef funct::Sum<Sig1, Bkg1>::type Fun1;
typedef funct::Sum<Sig2, Bkg2>::type Fun2;
typedef fit::MultiHistoChiSquare<Fun1, Fun2> ChiSquared;

int main(int ac, char *av[]) {
  gROOT->SetStyle("Plain");
  try {
    double fMin, fMax;
    string ext;
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help,h", "produce help message")
      ("input-file,i", po::value< vector<string> >(), "input file")
      ("min,m", po::value<double>(&fMin)->default_value(60), "minimum value for fit range")
      ("max,M", po::value<double>(&fMax)->default_value(120), "maximum value for fit range")
      ("plot-format,p", po::value<string>(&ext)->default_value("ps"), 
       "output plot format")
      ;
    
    po::positional_options_description p;
    p.add("input-file", -1);
    
    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).
	    options(desc).positional(p).run(), vm);
    po::notify(vm);
    
    if (vm.count("help")) {
      cout << "Usage: options_description [options]\n";
      cout << desc;
      return 0;
      }
    
    fit::RootMinuitCommands<ChiSquared> commands("csa08IsoBkg.txt");

    if (vm.count("input-file")) {
      cout << "Input files are: " 
	   << vm["input-file"].as< vector<string> >() << "\n";
      vector<string> v_file = vm["input-file"].as< vector<string> >();
      for(vector<string>::const_iterator it = v_file.begin(); 
	  it != v_file.end(); ++it) {
	TFile * root_file = new TFile(it->c_str(),"read");

	TH1D * histo1 = (TH1D*) root_file->Get("oneNonIsolatedZToMuMuPlots/zMass");
	fix(histo1);
	TH1D * histo2 = (TH1D*) root_file->Get("twoNonIsolatedZToMuMuPlots/zMass");
	fix(histo2);

	cout << ">>> histogram loaded\n";
	string f_string = *it;
	replace(f_string.begin(), f_string.end(), '.', '_');
	replace(f_string.begin(), f_string.end(), '/', '_');
	string plot_string = f_string + "." + ext;
	cout << ">>> Input files loaded\n";
	
	const char * kYieldZMuMu1 = "YieldZMuMu1";
	const char * kYieldZMuMu2 = "YieldZMuMu2";
	const char * kYieldBkg = "YieldBkg"; 
	const char * kEffBkg = "EffBkg"; 
	const char * kLambdaZMuMu = "LambdaZMuMu";
	const char * kMass = "Mass";
	const char * kGamma = "Gamma";
	const char * kPhotonFactorZMuMu = "PhotonFactorZMuMu";
	const char * kInterferenceFactorZMuMu = "InterferenceFactorZMuMu";
	const char * kMeanZMuMu = "MeanZMuMu";
	const char * kSigmaZMuMu = "SigmaZMuMu";
	const char * kAlpha = "Alpha";
	const char * kA0 = "A0"; 
	const char * kA1 = "A1"; 
	const char * kA2 = "A2"; 
	
	funct::Parameter lambdaZMuMu(kLambdaZMuMu, commands.par(kLambdaZMuMu));
	funct::Parameter mass(kMass, commands.par(kMass));
	funct::Parameter gamma(kGamma, commands.par(kGamma));
	funct::Parameter photonFactorZMuMu(kPhotonFactorZMuMu, commands.par(kPhotonFactorZMuMu)); 
	funct::Parameter interferenceFactorZMuMu(kInterferenceFactorZMuMu, commands.par(kInterferenceFactorZMuMu)); 
	funct::Parameter yieldZMuMu1(kYieldZMuMu1, commands.par(kYieldZMuMu1));
	funct::Parameter yieldZMuMu2(kYieldZMuMu2, commands.par(kYieldZMuMu2));
	funct::Parameter yieldBkg(kYieldBkg, commands.par(kYieldBkg));
	funct::Parameter effBkg(kEffBkg, commands.par(kEffBkg));
	funct::Parameter meanZMuMu(kMeanZMuMu, commands.par(kMeanZMuMu));
	funct::Parameter sigmaZMuMu(kSigmaZMuMu, commands.par(kSigmaZMuMu)); 
	funct::Parameter alpha(kAlpha, commands.par(kAlpha));
	funct::Parameter a0(kA0, commands.par(kA0));
	funct::Parameter a1(kA1, commands.par(kA1));
	funct::Parameter a2(kA2, commands.par(kA2));
	funct::Constant cFMin(fMin), cFMax(fMax);

	IntegratorConv integratorConv(1.e-4);
	IntegratorNorm integratorNorm(1.e-4);

	ZPeak zPeak = funct::Exponential(lambdaZMuMu) * 
	  funct::conv(funct::ZLineShape(mass, gamma, photonFactorZMuMu, interferenceFactorZMuMu), 
		      funct::Gaussian(meanZMuMu, sigmaZMuMu), 
		      -3*sigmaZMuMu.value(), 3*sigmaZMuMu.value(), integratorConv);
	SigPeak sp = funct::master(zPeak);
	SigPeakClone spc = funct::slave(sp);
	Sig1 sig1 = yieldZMuMu1 * sp;
	Sig2 sig2 = yieldZMuMu2 * spc;
	funct::Numerical<1> _1;
	funct::Numerical<2> _2;
	ExpPoly ep = funct::Exponential(alpha) * funct::Polynomial<2>(a0, a1, a2);
	ExpPolyNorm epn = ep / ExpPolyNormFactor(ep, cFMin, cFMax, integratorNorm);
	Coeff1 c1 = yieldBkg * (_2 / effBkg - _2);
	Bkg1 bkg1 = c1 * epn;
	Coeff2 c2 = yieldBkg * ((_1 /effBkg - _1) ^ _2);
	Bkg2 bkg2 = c2 * epn;
	Fun1 f1 = sig1 + bkg1;
	Fun2 f2 = sig2 + bkg2;

	ChiSquared chi2(f1, histo1, f2, histo2, fMin, fMax);
	cout << "N. deg. of freedom: " << chi2.degreesOfFreedom() << endl;
	fit::RootMinuit<ChiSquared> minuit(chi2, true);
	commands.add(minuit, yieldZMuMu1);
	commands.add(minuit, yieldZMuMu2);
	commands.add(minuit, yieldBkg);
	commands.add(minuit, effBkg);
	commands.add(minuit, lambdaZMuMu);
	commands.add(minuit, mass);
	commands.add(minuit, gamma);
	commands.add(minuit, photonFactorZMuMu);
	commands.add(minuit, interferenceFactorZMuMu);
	commands.add(minuit, meanZMuMu);
	commands.add(minuit, sigmaZMuMu);
	commands.add(minuit, alpha);
	commands.add(minuit, a0);
	commands.add(minuit, a1);
	commands.add(minuit, a2);
	commands.run(minuit);
	const unsigned int nPar = 15;//WARNIG: this must be updated manually for now
	ROOT::Math::SMatrix<double, nPar, nPar, ROOT::Math::MatRepSym<double, nPar> > err;
	minuit.getErrorMatrix(err);
	std::cout << "error matrix:" << std::endl;
	for(unsigned int i = 0; i < nPar; ++i) {
	  for(unsigned int j = 0; j < nPar; ++j) {
	    std::cout << err(i, j) << "\t";
	  }
	  std::cout << std::endl;
	} 
	minuit.printFitResults();

	funct::GaussIntegrator integrator(1.e-6);
	double nbkg1 = funct::integral_f(bkg1, fMin, fMax, integrator);
	double nbkg2 = funct::integral_f(bkg2, fMin, fMax, integrator);
	std::cout << "Background yields in [" << fMin <<", " << fMax << "]: "
		  << nbkg1 <<", " <<nbkg2 << std::endl;


	// binning is at 1 GeV, fortunately
	double i1 = histo1->Integral(int(fMin), int(fMax));
	double i2 = histo2->Integral(int(fMin), int(fMax));
	std::cout << "Histogram integrals in [" << fMin <<", " << fMax << "]: "
		  << i1 <<", " << i2 << std::endl;
	double s;
	s = 0;
	for(int i = 1; i <= histo1->GetNbinsX(); ++i)
	  s += histo1->GetBinContent(i);
	histo1->SetEntries(s);
	s = 0;
	for(int i = 1; i <= histo2->GetNbinsX(); ++i)
	  s += histo2->GetBinContent(i);
	histo2->SetEntries(s);

	double extrap = nbkg1*nbkg1 / nbkg2 /4;
	cout << "extrapolated background with no isolated muons:" << extrap << endl;

	string Plot1 = "OneIsolated_" + plot_string;
	root::plot<Fun1>(Plot1.c_str(), *histo1, f1, fMin, fMax, 
			 yieldZMuMu1, lambdaZMuMu, mass, gamma, photonFactorZMuMu, interferenceFactorZMuMu, 
			 meanZMuMu, sigmaZMuMu, yieldBkg, effBkg, alpha, a0, a1, a2,
			 kRed, 2, kDashed, 100, 
			 "Z -> #mu #mu mass", "#mu #mu invariant mass (GeV/c^{2})", 
			 "Events");
	string Plot2 = "TwoIsolated_" + plot_string;
	root::plot<Fun2>(Plot2.c_str(), *histo2, f2, fMin, fMax, 
			 yieldZMuMu2, lambdaZMuMu, mass, gamma, photonFactorZMuMu, interferenceFactorZMuMu, 
			 meanZMuMu, sigmaZMuMu, yieldBkg, effBkg, alpha, a0, a1, a2,
			 kRed, 2, kDashed, 100, 
			 "Z -> #mu #mu mass", "#mu #mu invariant mass (GeV/c^{2})", 
			 "Events");
      }
    }
    
  }
  catch(exception& e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
  }
  catch(...) {
    cerr << "Exception of unknown type!\n";
  }
  return 0;
}



