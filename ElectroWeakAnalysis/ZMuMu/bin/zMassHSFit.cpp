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
#include "TFile.h"
#include "TH1.h"
#include "TF1.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TROOT.h"
#include <boost/shared_ptr.hpp>
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
using namespace ::function;

// A helper function to simplify the main part.
template<class T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
    copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
    return os;
}

int main(int ac, char *av[]) { 
  try {
      double fMin, fMax;
      string ext;
      po::options_description desc("Allowed options");
      desc.add_options()
	("help", "produce help message")
	("include-path,I", po::value< vector<string> >(), 
	 "include path")
	("input-file", po::value< vector<string> >(), "input file")
	("min,m", po::value<double>(&fMin)->default_value(60), "minimum value for fit range")
	("max,M", po::value<double>(&fMax)->default_value(120), "maximum value for fit range")
	("convbwintgamg", 
	 "fit to the convolution of a breit-wigner plus interference term and gamma propagator and a gaussian")
	("convexpbwintgamg", 
	 "fit to the convolution of the product between an exponential and a breit-wigner plus interference term and gamma propagator with a gaussian")
	("convbwintgam2gf", 
	 "fit to the convolution of the breit-wigner plus interference term and gamma propagator with a linear combination of fixed gaussians")
	("output-file,O", po::value<string>(&ext)->default_value(".ps"), 
	 "output file format")
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
      
      if (vm.count("include-path"))
        {
	  cout << "Include paths are: " 
	       << vm["include-path"].as< vector<string> >() << "\n";
        }
      
      vector<string> v_file;
      vector<TH1D*> v_ZMassHistos;
      vector<string> v_eps;
      
      if (vm.count("input-file"))
	{
	  cout << "Input files are: " 
	       << vm["input-file"].as< vector<string> >() << "\n";
	  v_file = vm["input-file"].as< vector<string> >();
	  for(vector<string>::const_iterator it = v_file.begin(); 
	      it != v_file.end(); ++it) { 
	     TFile * root_file = new TFile(it->c_str(),"read");
	     TDirectory *Histos = (TDirectory*) root_file->GetDirectory("ZHisto");
	     TDirectory *RecoHistos = (TDirectory*) Histos->GetDirectory("ZRecoHisto");
	     TH1D * zMass = (TH1D*) RecoHistos->Get("ZMass");
	     zMass->Rebin(4); //remember...
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
	  cout <<">>> Input files loaded\n";
	}
      //Values for Z mass and width
      Parameter mass("Mass", 91.364);
      Parameter gamma("Gamma", 4.11);
      Parameter dmass("Mass Error", 0.0011);
      Parameter dgamma("Gamma Error", 0.004);
      //Parameters for Z Line Shape
      Parameter f_gamma("Photon factor", 0.838);
      Parameter f_int("Interference factor", -0.00197);
      Parameter df_gamma("Photon factor Error", 0.005);
      Parameter df_int("Interference factor Error", 0.00008);
      //Parameters for fits with gaussians
      Parameter yield("Yield", 283000);
      Parameter alpha("Alpha", 0.771); //the first gaussian is narrow
      Parameter mean("Mean", 0); //0.229
      Parameter sigma1("Sigma 1", 0.76); 
      Parameter sigma2("Sigma 2", 2.94);
      Parameter dyield("Yield Error", 9000);
      Parameter dalpha("Alpha Error", 0.005);
      Parameter dmean("Mean Error", 0); //0.005
      Parameter dsigma1("Sigma 1 Error", 0.04);
      Parameter dsigma2("Sigma 2 Error", 0.03);
      //Parameter for exponential
      Parameter lambda("Lambda", 0);
      Parameter dlambda("Lambda Error", 0);
      
      if (vm.count("convbwintgamg")) {
	cout << "Fitting histograms in input files to the convolution of the Breit-Wigner plus Z/photon interference and photon propagator with a Gaussian\n";
	cout << ">>> set pars: " << endl;
	cout << yield << " ; " << dyield << endl; 
	cout << mass << " ; " << dmass << endl; 
	cout << gamma << " ; " << dgamma << endl; 
	cout << f_gamma << " ; " << df_gamma << endl; 
	cout << f_int << " ; " << df_int << endl; 
	cout << mean << " ; " << dmean << endl; 
	cout << sigma1 << " ; " << dsigma1 << endl;
	for(size_t i = 0; i < v_ZMassHistos.size(); ++i) { 
	  TH1D * zMass = v_ZMassHistos[i]; 
	  ZLineShape zls(mass, gamma, f_gamma, f_int);
	  Gaussian gauss(mean, sigma1);
	  double range = 3 * sigma1.value();
	  Convolution<ZLineShape, Gaussian> czg(zls, gauss, -range , range, 1000);
	  Constant c(yield);
	  typedef Product<Constant, Convolution<ZLineShape, Gaussian> > FitFunction;
	  FitFunction f = c * czg;
	  cout << "set functions" << endl;
	  typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	  ChiSquared chi2(f, zMass, fMin, fMax);
	  int fullBins = chi2.degreesOfFreedom();
	  cout << "N. deg. of freedom: " << fullBins << endl;
	  fit::RootMinuit<ChiSquared> minuit(7, chi2, true);
	  minuit.setParameter(0, yield, 10, 100, 10000000);
	  minuit.setParameter(1, mass, .1, 70., 110);
	  minuit.setParameter(2, gamma, 1, 1, 10);
	  minuit.setParameter(3, f_gamma, 0.1, -100, 1000);
	  minuit.setParameter(4, f_int, .0001, -1000000, 1000000);
	  minuit.setParameter(5, mean, 0.001, -0.5, 0.5);
	  minuit.fixParameter(5);
	  minuit.setParameter(6, sigma1, 0.1, -5., 5.);
	  double amin = minuit.minimize();
	  cout << "fullBins = " << fullBins 
	       << "; free pars = " << minuit.getNumberOfFreeParameters() 
	       << endl;
	  unsigned int ndof = fullBins - minuit.getNumberOfFreeParameters();
	  cout << "Chi^2 = " << amin << "/" << ndof << " = " << amin/ndof 
	       << "; prob: " << TMath::Prob( amin, ndof )
	       << endl;
	  dyield = minuit.getParameterError(0);
	  cout << yield << " ; " << dyield << endl;
	  dmass = minuit.getParameterError(1);
	  cout << mass << " ; " << dmass << endl;
	  dgamma = minuit.getParameterError(2);
	  cout << gamma << " ; " << dgamma << endl;
	  df_gamma = minuit.getParameterError(3);
	  cout << f_gamma << " ; " << df_gamma << endl;
	  df_int = minuit.getParameterError(4);
	  cout << f_int << " ; " << df_int << endl;
	  //dmean = minuit.getParameterError(5);
	  cout << mean << " ; " << dmean << endl;
	  dsigma1 = minuit.getParameterError(6);
	  cout << sigma1 << " ; " << dsigma1 << endl;
	  vector<shared_ptr<double> > pars;
	  pars.push_back(yield.ptr());
	  pars.push_back(mass.ptr());
	  pars.push_back(gamma.ptr());
	  pars.push_back(f_gamma.ptr());
	  pars.push_back(f_int.ptr());
	  pars.push_back(mean.ptr());
	  pars.push_back(sigma1.ptr());
	  TF1 fun = root::tf1("fun", f, fMin, fMax, pars);
	  fun.SetParNames(yield.name().c_str(), mass.name().c_str(), gamma.name().c_str(), 
			  f_gamma.name().c_str(), f_int.name().c_str(), 
			  mean.name().c_str(), sigma1.name().c_str());
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
	cout << "Fitting histograms in input files to the convolution of the product between an exponential and a breit-wigner plus interference term and gamma propagator with a gaussian" 
	     << endl;
	cout << ">>> set pars: " << endl;
	cout << yield << " ; " << dyield << endl; 
	cout << lambda << " ; " << dlambda << endl;
	cout << mass << " ; " << dmass << endl; 
	cout << gamma << " ; " << dgamma << endl; 
	cout << f_gamma << " ; " << df_gamma << endl; 
	cout << f_int << " ; " << df_int << endl; 
	cout << mean << " ; " << dmean << endl; 
	cout << sigma1 << " ; " << dsigma1 << endl;
	for(size_t i = 0; i < v_ZMassHistos.size(); ++i) { 
	  TH1D * zMass = v_ZMassHistos[i]; 
	  Exponential expo(lambda);
	  ZLineShape zls(mass, gamma, f_gamma, f_int);
	  Gaussian gauss(mean, sigma1);
	  typedef Product<Exponential, ZLineShape> ExpZLS;
	  ExpZLS expz = expo * zls;
	  double range = 3 * sigma1.value();
	  Convolution<ExpZLS, Gaussian> cezg(expz, gauss, -range , range, 1000);
	  Constant c(yield);
	  typedef Product<Constant, Convolution<ExpZLS, Gaussian> > FitFunction;
	  FitFunction f = c * cezg;
	  cout << "set functions" << endl;
	  typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	  ChiSquared chi2(f, zMass, fMin, fMax);
	  int fullBins = chi2.degreesOfFreedom();
	  cout << "N. deg. of freedom: " << fullBins << endl;
	  fit::RootMinuit<ChiSquared> minuit(8, chi2, true);
	  minuit.setParameter(0, yield, 10, 100, 10000000);
	  minuit.setParameter(1, lambda, 0.1, -100, 100);
	  minuit.fixParameter(1);
	  minuit.setParameter(2, mass, .1, 70., 110);
	  minuit.setParameter(3, gamma, 1, 1, 10);
	  minuit.setParameter(4, f_gamma, 0.1, -100, 1000);
	  minuit.setParameter(5, f_int, .0001, -1000000, 1000000);
	  minuit.setParameter(6, mean, 0.001, -0.5, 0.5);
	  minuit.fixParameter(6);
	  minuit.setParameter(7, sigma1, 0.1, -5., 5.);
	  double amin = minuit.minimize();
	  cout << "fullBins = " << fullBins 
	       << "; free pars = " << minuit.getNumberOfFreeParameters() 
	       << endl;
	  unsigned int ndof = fullBins - minuit.getNumberOfFreeParameters();
	  cout << "Chi^2 = " << amin << "/" << ndof << " = " << amin/ndof 
	       << "; prob: " << TMath::Prob( amin, ndof )
	       << endl;
	  dyield = minuit.getParameterError(0);
	  cout << yield << " ; " << dyield << endl;
	  dlambda = minuit.getParameterError(1);
	  cout << lambda << " ; " << dlambda << endl;
	  dmass = minuit.getParameterError(2);
	  cout << mass << " ; " << dmass << endl;
	  dgamma = minuit.getParameterError(3);
	  cout << gamma << " ; " << dgamma << endl;
	  df_gamma = minuit.getParameterError(4);
	  cout << f_gamma << " ; " << df_gamma << endl;
	  df_int = minuit.getParameterError(5);
	  cout << f_int << " ; " << df_int << endl;
	  //dmean = minuit.getParameterError(6);
	  cout << mean << " ; " << dmean << endl;
	  dsigma1 = minuit.getParameterError(7);
	  cout << sigma1 << " ; " << dsigma1 << endl;
	  minuit.releaseParameter(1);
	  amin = minuit.minimize();
	  cout << "fullBins = " << fullBins 
	       << "; free pars = " << minuit.getNumberOfFreeParameters() 
	       << endl;
	  ndof = fullBins - minuit.getNumberOfFreeParameters();
	  cout << "Chi^2 = " << amin << "/" << ndof << " = " << amin/ndof 
	       << "; prob: " << TMath::Prob( amin, ndof )
	       << endl;
	  dyield = minuit.getParameterError(0);
	  cout << yield << " ; " << dyield << endl;
	  dlambda = minuit.getParameterError(1);
	  cout << lambda << " ; " << dlambda << endl;
	  dmass = minuit.getParameterError(2);
	  cout << mass << " ; " << dmass << endl;
	  dgamma = minuit.getParameterError(3);
	  cout << gamma << " ; " << dgamma << endl;
	  df_gamma = minuit.getParameterError(4);
	  cout << f_gamma << " ; " << df_gamma << endl;
	  df_int = minuit.getParameterError(5);
	  cout << f_int << " ; " << df_int << endl;
	  //dmean = minuit.getParameterError(6);
	  cout << mean << " ; " << dmean << endl;
	  dsigma1 = minuit.getParameterError(7);
	  cout << sigma1 << " ; " << dsigma1 << endl;
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
	  fun.SetParNames(yield.name().c_str(), lambda.name().c_str(),
			  mass.name().c_str(), gamma.name().c_str(), 
			  f_gamma.name().c_str(), f_int.name().c_str(), 
			  mean.name().c_str(), sigma1.name().c_str());
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
	cout << "Fitting histograms in input files to the convolution of the Breit-Wigner plus Z/photon interference and photon propagator with a linear combination of fixed Gaussians\n";
	cout << ">>> set pars: " << endl;
	cout << yield << " ; " << dyield << endl; 
	cout << alpha << " ; " << dalpha << endl; 
	cout << mass << " ; " << dmass << endl; 
	cout << gamma << " ; " << dgamma << endl; 
	cout << f_gamma << " ; " << df_gamma << endl; 
	cout << f_int << " ; " << df_int << endl; 
	cout << mean << " ; " << dmean << endl; 
	cout << sigma1 << " ; " << dsigma1 << endl; 
	cout << sigma2 << " ; " << dsigma2 << endl;
	for(size_t i = 0; i < v_ZMassHistos.size(); ++i) { 
	  TH1D * zMass = v_ZMassHistos[i];
	  ZLineShape zls(mass, gamma, f_gamma, f_int);
	  Gaussian gaus1(mean, sigma1);
	  Gaussian gaus2(mean, sigma2);
	  Number _1(1);
	  typedef Product<Constant, Gaussian> G1;
	  typedef Product<Difference<Number,Constant>, Gaussian> G2;
	  typedef Product<Constant,Sum<G1, G2> > GaussComb;
	  GaussComb gc = yield*(alpha*gaus1 + (_1 - alpha)*gaus2);
	  typedef Convolution<ZLineShape, GaussComb> FitFunction;
	  double range = 3 * max(sigma1.value(), sigma2.value());
	  FitFunction f(zls, gc, -range , range, 1000);
	  cout << "set functions" << endl;
	  typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	  ChiSquared chi2(f, zMass, fMin, fMax);
	  int fullBins = chi2.degreesOfFreedom();
	  cout << "N. deg. of freedom: " << fullBins << endl;
	  fit::RootMinuit<ChiSquared> minuit(9, chi2, true);
	  minuit.setParameter(0, yield, 10, 100, 10000000);
	  //minuit.fixParameter(0);
	  minuit.setParameter(1, alpha, 0.1, -1., 1.);
	  //minuit.fixParameter(1);
	  minuit.setParameter(2, mass, .1, 70., 110);
	  //minuit.fixParameter(2);
	  minuit.setParameter(3, gamma, 1, 1, 10);
	  //minuit.fixParameter(3);
	  minuit.setParameter(4, f_gamma, 0.1, -100, 1000);
	  //minuit.fixParameter(4);
	  minuit.setParameter(5, f_int, .0001, -1000000, 1000000);
	  //minuit.fixParameter(5);
	  minuit.setParameter(6, mean, 0.001, -0.5, 0.5);
	  minuit.fixParameter(6);
	  minuit.setParameter(7, sigma1, 0.1, -5., 5.);
	  //	    minuit.fixParameter(7);
	  minuit.setParameter(8, sigma2, 0.1, -5., 5.);
	  //	    minuit.fixParameter(8);
	  double amin = minuit.minimize();
	  cout << "fullBins = " << fullBins 
	       << "; free pars = " << minuit.getNumberOfFreeParameters() 
	       << endl;
	  unsigned int ndof = fullBins - minuit.getNumberOfFreeParameters();
	  cout << "Chi^2 = " << amin << "/" << ndof << " = " << amin/ndof 
	       << "; prob: " << TMath::Prob( amin, ndof )
	       << endl;
	  dyield = minuit.getParameterError(0);
	  cout << yield << " ; " << dyield << endl;
	  dalpha = minuit.getParameterError(1);
	  cout << alpha << " ; " << dalpha << endl;
	  dmass = minuit.getParameterError(2);
	  cout << mass << " ; " << dmass << endl;
	  dgamma = minuit.getParameterError(3);
	  cout << gamma << " ; " << dgamma << endl;
	  df_gamma = minuit.getParameterError(4);
	  cout << f_gamma << " ; " << df_gamma << endl;
	  df_int = minuit.getParameterError(5);
	  cout << f_int << " ; " << df_int << endl;
	  //dmean = minuit.getParameterError(6);
	  cout << mean << " ; " << dmean << endl;
	  dsigma1 = minuit.getParameterError(7);
	  cout << sigma1 << " ; " << dsigma1 << endl;
	  dsigma2 = minuit.getParameterError(8);
	  cout << sigma2 << " ; " << dsigma2 << endl;
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
	  fun.SetParNames(yield.name().c_str(), alpha.name().c_str(), 
			  mass.name().c_str(), gamma.name().c_str(), 
			  f_gamma.name().c_str(), f_int.name().c_str(), 
			  mean.name().c_str(), sigma1.name().c_str(), sigma2.name().c_str());
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
