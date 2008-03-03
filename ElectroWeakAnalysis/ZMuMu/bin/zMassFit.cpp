#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "PhysicsTools/Utilities/interface/BreitWigner.h"
#include "PhysicsTools/Utilities/interface/Constant.h"
#include "PhysicsTools/Utilities/interface/Product.h"
#include "PhysicsTools/Utilities/interface/Sum.h"
#include "PhysicsTools/Utilities/interface/Gaussian.h"
#include "PhysicsTools/Utilities/interface/ZLineShape.h"
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
//#include "TStyle.h"
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
	("min,m", po::value<double>(&fMin)->default_value(80), "minimum value for fit range")
	("max,M", po::value<double>(&fMax)->default_value(120), "maximum value for fit range")
	("breitwigner", "fit to a breit-wigner")
	("gauss", "fit to a gaussian")
	("bwint", "fit to the breit-wigner plus interference term")
	("bwintgam", "fit to the breit-wigner plus interference term and gamma propagator")
	("convbwg", "fit to the convolution between a breit-wigner and a gaussian")
	("convbwintg", "fit to the convolution between a breit-wigner plus interference term and a gaussian")
	("convbwintgamg", "fit to the convolution of a breit-wigner plus interference term and gamma propagator and a gaussian")
	("convbw2gf", "fit to the convolution between a breit-wigner and a linear combination of fixed gaussians")
	("convbwf2g", "fit to the convolution between a fixed breit-wigner and a linear combination of gaussians")
	("convbwint2gf", "fit to the convolution between the breit-wigner plus interference term and a linear combination of fixed gaussians")
	("convbwintf2g", "fit to the convolution between the fixed breit-wigner plus interference term and a linear combination of gaussians")
	("convbwintgam2gf", "fit to the convolution of the breit-wigner plus interference term and gamma propagator with a linear combination of fixed gaussians")
	("convbwintgamf2g", "fit to the convolution of the fixed breit-wigner plus interference term and gamma propagator with a linear combination of gaussians")
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
	     TH1D * zMass = (TH1D*) RecoHistos->Get("MuMuMass");
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
      //PDG values for Z mass and width
      Parameter mass("Mass", 91.1876);
      Parameter gamma("Gamma", 2.4952);
      Parameter dmass("Mass Error", 0.0021);
      Parameter dgamma("Gamma Error", 0.0023);
      //Parameters for Z Line Shape
      Parameter f_gamma("Photon factor", 0);
      Parameter f_int("Interference factor", 0.001);
      Parameter df_gamma("Photon factor Error", 0);
      Parameter df_int("Interference factor Error", 0);
      //Parameters for fits with one gaussian
      Parameter yield("Yield", 1000);
      Parameter mean("Mean", 0);
      Parameter sigma("Sigma", 1.);
      Parameter dyield("Yield Error", 0);
      Parameter dmean("Mean Error", 0);
      Parameter dsigma("Sigma Error", 0);
      
      if (vm.count("breitwigner"))
	{
	  cout << "Fitting histograms in input file to a Breit-Wigner\n";
	  cout << ">>> set pars: " << endl;
	  cout << yield << " ; " << dyield << endl; 
	  cout << mass << " ; " << dmass << endl; 
	  cout << gamma << " ; " << dgamma << endl; 
	  for(size_t i = 0; i < v_ZMassHistos.size(); ++i) { 
	    cout << ">>> load histogram\n";
	    TH1D * zMass = v_ZMassHistos[i];
	    cout << ">>> histogram loaded\n";
	    BreitWigner bw(mass, gamma);
	    Constant c(yield);
	    typedef Product<Constant, BreitWigner> FitFunction;
	    FitFunction f = c * bw;
	    typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	    ChiSquared chi2(f, zMass, fMin, fMax);
	    int fullBins = chi2.degreesOfFreedom();
	    cout << "N. deg. of freedom: " << fullBins << endl;
	    fit::RootMinuit<ChiSquared> minuit(3, chi2, true);
	    minuit.setParameter(0, yield, 10, 100, 100000);
	    minuit.setParameter(1, mass, .1, 70., 110);
	    minuit.setParameter(2, gamma, 1, 1, 10);
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
	    TF1 fun = root::tf1("fun", f, fMin, fMax, yield, mass, gamma);
	    fun.SetParNames(yield.name().c_str(), mass.name().c_str(), gamma.name().c_str());
	    fun.SetLineColor(kRed);
	    TCanvas *canvas = new TCanvas("canvas");
	    zMass->Draw("e");
	    fun.Draw("same");	
	    string epsFilename = "ZMassFitBW_" + v_eps[i];
	    canvas->SaveAs(epsFilename.c_str());
	    canvas->SetLogy();
	    string epsLogFilename = "ZMassFitBW_Log_" + v_eps[i];
	    canvas->SaveAs(epsLogFilename.c_str());
	  }
	}
      
      if (vm.count("gauss"))
	{
	  cout << "Fitting histograms in input files to a Gaussian\n"; 
	  cout << ">>> set pars: " << endl;
	  cout << yield << " ; " << dyield << endl; 
	  cout << mean << " ; " << dmean << endl; 
	  cout << sigma << " ; " << dsigma << endl;
	  for(size_t i = 0; i < v_ZMassHistos.size(); ++i) { 
	    TH1D * zMass = v_ZMassHistos[i]; 
	    Gaussian gaus(mean, sigma);
	    Constant c(yield);
	    typedef Product<Constant, Gaussian> FitFunction;
	    FitFunction f = c * gaus;
	    typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	    ChiSquared chi2(f, zMass, fMin, fMax);
	    int fullBins = chi2.degreesOfFreedom();
	    cout << "N. deg. of freedom: " << fullBins << endl;
	    fit::RootMinuit<ChiSquared> minuit(3, chi2, true);
	    minuit.setParameter(0, yield, 10, 100, 100000);
	    minuit.setParameter(1, mean, 0.001, 80, 100);
	    minuit.setParameter(2, sigma, 0.1, -5., 5.);
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
	    dmean = minuit.getParameterError(1);
	    cout << mean << " ; " << dmean << endl;
	    dsigma = minuit.getParameterError(2);
	    cout << sigma << " ; " << dsigma << endl;
	    TF1 fun = root::tf1("fun", f, fMin, fMax, yield, mean, sigma);
	    fun.SetParNames(yield.name().c_str(), mean.name().c_str(), sigma.name().c_str());
	    fun.SetLineColor(kRed);
	    TCanvas *canvas = new TCanvas("canvas");
	    zMass->Draw("e");
	    fun.Draw("same");	
	    string epsFilename = "ZMassFitG_" + v_eps[i];
	    canvas->SaveAs(epsFilename.c_str());
	    canvas->SetLogy();
	    string epsLogFilename = "ZMassFitG_Log_" + v_eps[i];
	    canvas->SaveAs(epsLogFilename.c_str());
	  }
	}
      
      if (vm.count("bwint"))
	{
	  cout << "Fitting histograms in input files to the Breit-Wigner plus Z/photon interference term\n";
	  cout << ">>> set pars: " << endl;
	  cout << yield << " ; " << dyield << endl; 
	  cout << mass << " ; " << dmass << endl; 
	  cout << gamma << " ; " << dgamma << endl; 
	  cout << f_gamma << " ; " << df_gamma << endl; 
	  cout << f_int << " ; " << df_int << endl; 
	  for(size_t i = 0; i < v_ZMassHistos.size(); ++i) { 
	    TH1D * zMass = v_ZMassHistos[i]; 
	    ZLineShape zls(mass, gamma, f_gamma, f_int);
	    Constant c(yield);
	    typedef Product<Constant, ZLineShape> FitFunction;
	    FitFunction f = c * zls;
	    cout << "set functions" << endl;
	    vector<shared_ptr<double> > pars;
	    pars.push_back(yield.ptr());
	    pars.push_back(mass.ptr());
	    pars.push_back(gamma.ptr());
	    pars.push_back(f_gamma.ptr());
	    pars.push_back(f_int.ptr());
	    typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	    ChiSquared chi2(f, zMass, fMin, fMax);
	    int fullBins = chi2.degreesOfFreedom();
	    cout << "N. deg. of freedom: " << fullBins << endl;
	    fit::RootMinuit<ChiSquared> minuit(5, chi2, true);
	    minuit.setParameter(0, yield, 10, 100, 100000);
	    minuit.setParameter(1, mass, .1, 70., 110);
	    minuit.setParameter(2, gamma, 1, 1, 10);
	    minuit.setParameter(3, f_gamma, 0.1, -100, 1000);
	    minuit.fixParameter(3);
	    minuit.setParameter(4, f_int, .0001, -1000000, 1000000);
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
	    TF1 fun = root::tf1("fun", f, fMin, fMax, pars); 
	    fun.SetParNames(yield.name().c_str(), mass.name().c_str(), gamma.name().c_str(), 
	                    f_gamma.name().c_str(), f_int.name().c_str());
	    fun.SetLineColor(kRed); 
	    TCanvas *canvas = new TCanvas("canvas");
	    zMass->Draw("e");
	    fun.Draw("same");
	    string epsFilename = "ZMassFitBwIn_" + v_eps[i];
	    canvas->SaveAs(epsFilename.c_str());
	    canvas->SetLogy();
	    string epsLogFilename = "ZMassFitBwIn_Log_" + v_eps[i];
	    canvas->SaveAs(epsLogFilename.c_str());
	  }
	}
      
      if (vm.count("bwint"))
	{
	  cout << "Fitting histograms in input files to the Breit-Wigner plus Z/photon interference term and gamma propagator\n";
	  cout << ">>> set pars: " << endl;
	  cout << yield << " ; " << dyield << endl; 
	  cout << mass << " ; " << dmass << endl; 
	  cout << gamma << " ; " << dgamma << endl; 
	  cout << f_gamma << " ; " << df_gamma << endl; 
	  cout << f_int << " ; " << df_int << endl; 
	  for(size_t i = 0; i < v_ZMassHistos.size(); ++i) { 
	    TH1D * zMass = v_ZMassHistos[i]; 
	    ZLineShape zls(mass, gamma, f_gamma, f_int);
	    Constant c(yield);
	    typedef Product<Constant, ZLineShape> FitFunction;
	    FitFunction f = c * zls;
	    cout << "set functions" << endl;
	    vector<shared_ptr<double> > pars;
	    pars.push_back(yield.ptr());
	    pars.push_back(mass.ptr());
	    pars.push_back(gamma.ptr());
	    pars.push_back(f_gamma.ptr());
	    pars.push_back(f_int.ptr());
	    typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	    ChiSquared chi2(f, zMass, fMin, fMax);
	    int fullBins = chi2.degreesOfFreedom();
	    cout << "N. deg. of freedom: " << fullBins << endl;
	    fit::RootMinuit<ChiSquared> minuit(5, chi2, true);
	    minuit.setParameter(0, yield, 10, 100, 100000);
	    minuit.setParameter(1, mass, .1, 70., 110);
	    minuit.setParameter(2, gamma, 1, 1, 10);
	    minuit.setParameter(3, f_gamma, 0.1, -100, 1000);
	    minuit.setParameter(4, f_int, .0001, -1000000, 1000000);
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
	    TF1 fun = root::tf1("fun", f, fMin, fMax, pars); 
	    fun.SetParNames(yield.name().c_str(), mass.name().c_str(), gamma.name().c_str(), 
	                    f_gamma.name().c_str(), f_int.name().c_str());
	    fun.SetLineColor(kRed); 
	    TCanvas *canvas = new TCanvas("canvas");
	    zMass->Draw("e");
	    fun.Draw("same");
	    string epsFilename = "ZMassFitBwInGam_" + v_eps[i];
	    canvas->SaveAs(epsFilename.c_str());
	    canvas->SetLogy();
	    string epsLogFilename = "ZMassFitBwInGam_Log_" + v_eps[i];
	    canvas->SaveAs(epsLogFilename.c_str());
	  }
	}
      
      if (vm.count("convbwg"))
	{
	  cout << "Fitting histograms in input files to the convolution between a Breit Wigner and a Gaussian\n";
	  cout << ">>> set pars: " << endl;
	  cout << yield << " ; " << dyield << endl; 
	  cout << mass << " ; " << dmass << endl; 
	  cout << gamma << " ; " << dgamma << endl; 
	  cout << mean << " ; " << dmean << endl; 
	  cout << sigma << " ; " << dsigma << endl;
	  for(size_t i = 0; i < v_ZMassHistos.size(); ++i) { 
	    TH1D * zMass = v_ZMassHistos[i]; 
	    BreitWigner bw(mass, gamma);
	    Gaussian gauss(mean, sigma);
	    double range = 3 * sigma.value();
	    Convolution<BreitWigner, Gaussian> cbg(bw, gauss, -range , range, 200);
	    Constant c(yield);
	    typedef Product<Constant, Convolution<BreitWigner, Gaussian> > FitFunction;
	    FitFunction f = c * cbg;
	    cout << "set functions" << endl;
	    typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	    ChiSquared chi2(f, zMass, fMin, fMax);
	    int fullBins = chi2.degreesOfFreedom();
	    cout << "N. deg. of freedom: " << fullBins << endl;
	    fit::RootMinuit<ChiSquared> minuit(5, chi2, true);
	    minuit.setParameter(0, yield, 10, 100, 100000);
	    minuit.setParameter(1, mass, .1, 70., 110);
	    minuit.setParameter(2, gamma, 1, 1, 10);
	    minuit.setParameter(3, mean, 0.001, -0.5, 0.5);
	    minuit.fixParameter(3);
	    minuit.setParameter(4, sigma, 0.1, -5., 5.);
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
	    dmean = minuit.getParameterError(3);
	    cout << mean << " ; " << dmean << endl;
	    dsigma = minuit.getParameterError(4);
	    cout << sigma << " ; " << dsigma << endl;
	    vector<shared_ptr<double> > pars;
	    pars.push_back(yield.ptr());
	    pars.push_back(mass.ptr());
	    pars.push_back(gamma.ptr());
	    pars.push_back(mean.ptr());
	    pars.push_back(sigma.ptr());
	    TF1 fun = root::tf1("fun", f, fMin, fMax, pars);
	    fun.SetParNames(yield.name().c_str(), mass.name().c_str(), gamma.name().c_str(), 
			    mean.name().c_str(), sigma.name().c_str());
	    fun.SetLineColor(kRed);
	    TCanvas *canvas = new TCanvas("canvas");
	    zMass->Draw("e");
	    fun.Draw("same");
	    string epsFilename = "ZMassFitCoBwG_" + v_eps[i];
	    canvas->SaveAs(epsFilename.c_str());
	    canvas->SetLogy();
	    string epsLogFilename = "ZMassFitCoBwG_Log_" + v_eps[i];
	    canvas->SaveAs(epsLogFilename.c_str());
	  }
	}
      
      if (vm.count("convbwintg"))
	{
	  cout << "Fitting histograms in input files to the convolution between the Breit-Wigner plus Z/photon interference and a Gaussian\n";
	  cout << ">>> set pars: " << endl;
	  cout << yield << " ; " << dyield << endl; 
	  cout << mass << " ; " << dmass << endl; 
	  cout << gamma << " ; " << dgamma << endl; 
	  cout << f_gamma << " ; " << df_gamma << endl; 
	  cout << f_int << " ; " << df_int << endl; 
	  cout << mean << " ; " << dmean << endl; 
	  cout << sigma << " ; " << dsigma << endl;
	  for(size_t i = 0; i < v_ZMassHistos.size(); ++i) { 
	    TH1D * zMass = v_ZMassHistos[i]; 
	    ZLineShape zls(mass, gamma, f_gamma, f_int);
	    Gaussian gauss(mean, sigma);
	    double range = 3 * sigma.value();
	    Convolution<ZLineShape, Gaussian> czg(zls, gauss, -range , range, 200);
	    Constant c(yield);
	    typedef Product<Constant, Convolution<ZLineShape, Gaussian> > FitFunction;
	    FitFunction f = c * czg;
	    cout << "set functions" << endl;
	    typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	    ChiSquared chi2(f, zMass, fMin, fMax);
	    int fullBins = chi2.degreesOfFreedom();
	    cout << "N. deg. of freedom: " << fullBins << endl;
	    fit::RootMinuit<ChiSquared> minuit(7, chi2, true);
	    minuit.setParameter(0, yield, 10, 100, 100000);
	    minuit.setParameter(1, mass, .1, 70., 110);
	    minuit.setParameter(2, gamma, 1, 1, 10);
	    minuit.setParameter(3, f_gamma, 0.1, -100, 1000);
	    minuit.fixParameter(3);
	    minuit.setParameter(4, f_int, .0001, -1000000, 1000000);
	    minuit.setParameter(5, mean, 0.001, -0.5, 0.5);
	    minuit.fixParameter(5);
	    minuit.setParameter(6, sigma, 0.1, -5., 5.);
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
	    dmean = minuit.getParameterError(5);
	    cout << mean << " ; " << dmean << endl;
	    dsigma = minuit.getParameterError(6);
	    cout << sigma << " ; " << dsigma << endl;
	    vector<shared_ptr<double> > pars;
	    pars.push_back(yield.ptr());
	    pars.push_back(mass.ptr());
	    pars.push_back(gamma.ptr());
	    pars.push_back(f_gamma.ptr());
	    pars.push_back(f_int.ptr());
	    pars.push_back(mean.ptr());
	    pars.push_back(sigma.ptr());
	    TF1 fun = root::tf1("fun", f, fMin, fMax, pars);
	    fun.SetParNames(yield.name().c_str(), mass.name().c_str(), gamma.name().c_str(), 
			    f_gamma.name().c_str(), f_int.name().c_str(), 
			    mean.name().c_str(), sigma.name().c_str());
	    fun.SetLineColor(kRed);
	    TCanvas *canvas = new TCanvas("canvas");
	    zMass->Draw("e");
	    fun.Draw("same");
	    string epsFilename = "ZMassFitCoBwInG_" + v_eps[i];
	    canvas->SaveAs(epsFilename.c_str());
	    canvas->SetLogy();
	    string epsLogFilename = "ZMassFitCoBwInG_Log_" + v_eps[i];
	    canvas->SaveAs(epsLogFilename.c_str());
	  }
	}
      if (vm.count("convbwintgamg"))
	{
	  cout << "Fitting histograms in input files to the convolution of the Breit-Wigner plus Z/photon interference and photon propagator with a Gaussian\n";
	  cout << ">>> set pars: " << endl;
	  cout << yield << " ; " << dyield << endl; 
	  cout << mass << " ; " << dmass << endl; 
	  cout << gamma << " ; " << dgamma << endl; 
	  cout << f_gamma << " ; " << df_gamma << endl; 
	  cout << f_int << " ; " << df_int << endl; 
	  cout << mean << " ; " << dmean << endl; 
	  cout << sigma << " ; " << dsigma << endl;
	  for(size_t i = 0; i < v_ZMassHistos.size(); ++i) { 
	    TH1D * zMass = v_ZMassHistos[i]; 
	    ZLineShape zls(mass, gamma, f_gamma, f_int);
	    Gaussian gauss(mean, sigma);
	    double range = 3 * sigma.value();
	    Convolution<ZLineShape, Gaussian> czg(zls, gauss, -range , range, 200);
	    Constant c(yield);
	    typedef Product<Constant, Convolution<ZLineShape, Gaussian> > FitFunction;
	    FitFunction f = c * czg;
	    cout << "set functions" << endl;
	    typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	    ChiSquared chi2(f, zMass, fMin, fMax);
	    int fullBins = chi2.degreesOfFreedom();
	    cout << "N. deg. of freedom: " << fullBins << endl;
	    fit::RootMinuit<ChiSquared> minuit(7, chi2, true);
	    minuit.setParameter(0, yield, 10, 100, 100000);
	    minuit.setParameter(1, mass, .1, 70., 110);
	    minuit.setParameter(2, gamma, 1, 1, 10);
	    minuit.setParameter(3, f_gamma, 0.1, -100, 1000);
	    minuit.setParameter(4, f_int, .0001, -1000000, 1000000);
	    minuit.setParameter(5, mean, 0.001, -0.5, 0.5);
	    minuit.fixParameter(5);
	    minuit.setParameter(6, sigma, 0.1, -5., 5.);
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
	    dmean = minuit.getParameterError(5);
	    cout << mean << " ; " << dmean << endl;
	    dsigma = minuit.getParameterError(6);
	    cout << sigma << " ; " << dsigma << endl;
	    vector<shared_ptr<double> > pars;
	    pars.push_back(yield.ptr());
	    pars.push_back(mass.ptr());
	    pars.push_back(gamma.ptr());
	    pars.push_back(f_gamma.ptr());
	    pars.push_back(f_int.ptr());
	    pars.push_back(mean.ptr());
	    pars.push_back(sigma.ptr());
	    TF1 fun = root::tf1("fun", f, fMin, fMax, pars);
	    fun.SetParNames(yield.name().c_str(), mass.name().c_str(), gamma.name().c_str(), 
			    f_gamma.name().c_str(), f_int.name().c_str(), 
			    mean.name().c_str(), sigma.name().c_str());
	    fun.SetLineColor(kRed);
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
      
      //Parameters for fit with two gaussians
      Parameter yield1("Yield 1", 600);
      Parameter yield2("Yield 2", 220);
      Parameter mean1("Mean 1", 0.08);
      Parameter sigma1("Sigma 1", 1.11);
      Parameter mean2("Mean 2", -1.);
      Parameter sigma2("Sigma 2", 3.9);
      Parameter dyield1("Yield 1 Error", 40);
      Parameter dyield2("Yield 2 Error", 40);
      Parameter dmean1("Mean 1 Error", 0.04);
      Parameter dsigma1("Sigma 1 Error", 0.06);
      Parameter dmean2("Mean 2 Error", 0.3);
      Parameter dsigma2("Sigma 2 Error", 0.5);
      
      if (vm.count("convbw2gf"))
	{ 
	  cout << "Fitting histograms in input files to the convolution between the Z Breit-Wigner and a linear combination of fixed Gaussians\n";
	  cout << ">>> set pars: " << endl;
	  cout << yield1 << " ; " << dyield1 << endl; 
	  cout << yield2 << " ; " << dyield2 << endl; 
	  cout << mass << " ; " << dmass << endl; 
	  cout << gamma << " ; " << dgamma << endl; 
	  cout << mean1 << " ; " << dmean1 << endl; 
	  cout << sigma1 << " ; " << dsigma1 << endl; 
	  cout << mean2 << " ; " << dmean2 << endl; 
	  cout << sigma2 << " ; " << dsigma2 << endl;
	  for(size_t i = 0; i < v_ZMassHistos.size(); ++i) { 
	    TH1D * zMass = v_ZMassHistos[i]; 
	    Constant c1(yield1);
	    Constant c2(yield2);
	    BreitWigner bw(mass, gamma);
	    Gaussian gaus1(mean1, sigma1);
	    Gaussian gaus2(mean2, sigma2);
	    typedef Product<Constant, Gaussian> GaussProduct;
	    GaussProduct gp1 = c1 * gaus1;
	    GaussProduct gp2 = c2 * gaus2;
	    typedef Sum<GaussProduct, GaussProduct> GaussComb;
	    GaussComb gc = gp1 + gp2;
	    typedef Convolution<BreitWigner, GaussComb> FitFunction;
	    double range = 3 * max(sigma1.value(), sigma2.value());
	    FitFunction f(bw, gc, -range , range, 200);
	    cout << "set functions" << endl;
	    typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	    ChiSquared chi2(f, zMass, fMin, fMax);
	    int fullBins = chi2.degreesOfFreedom();
	    cout << "N. deg. of freedom: " << fullBins << endl;
	    fit::RootMinuit<ChiSquared> minuit(8, chi2, true);
	    minuit.setParameter(0, yield1, 10, 100, 100000);
	    //minuit.fixParameter(0);
	    minuit.setParameter(1, yield2, 10, 100, 100000);
	    //minuit.fixParameter(1);
	    minuit.setParameter(2, mass, .1, 70., 110);
	    //minuit.fixParameter(2);
	    minuit.setParameter(3, gamma, 1, 1, 10);
	    //minuit.fixParameter(3);
	    minuit.setParameter(4, mean1, 0.001, -0.5, 0.5);
	    minuit.fixParameter(4);
	    minuit.setParameter(5, sigma1, 0.1, -5., 5.);
	    minuit.fixParameter(5);
	    minuit.setParameter(6, mean2, 0.001, -5., 5.);
	    minuit.fixParameter(6);
	    minuit.setParameter(7, sigma2, 0.1, -5., 5.); 
	    minuit.fixParameter(7);
	    double amin = minuit.minimize();
	    cout << "fullBins = " << fullBins 
		 << "; free pars = " << minuit.getNumberOfFreeParameters() 
		 << endl;
	    unsigned int ndof = fullBins - minuit.getNumberOfFreeParameters();
	    cout << "Chi^2 = " << amin << "/" << ndof << " = " << amin/ndof 
		 << "; prob: " << TMath::Prob( amin, ndof )
		 << endl;
	    dyield1 = minuit.getParameterError(0);
	    cout << yield1 << " ; " << dyield1 << endl;
	    dyield2 = minuit.getParameterError(1);
	    cout << yield2 << " ; " << dyield2 << endl;
	    dmass = minuit.getParameterError(2);
	    cout << mass << " ; " << dmass << endl;
	    dgamma = minuit.getParameterError(3);
	    cout << gamma << " ; " << dgamma << endl;
	    //dmean1 = minuit.getParameterError(4);
	    cout << mean1 << " ; " << dmean1 << endl;
	    //dsigma1 = minuit.getParameterError(5);
	    cout << sigma1 << " ; " << dsigma1 << endl;
	    //dmean2 = minuit.getParameterError(6);
	    cout << mean2 << " ; " << dmean2 << endl;
	    //dsigma2 = minuit.getParameterError(7);
	    cout << sigma2 << " ; " << dsigma2 << endl;
	    vector<shared_ptr<double> > pars;
	    pars.push_back(yield1.ptr());
	    pars.push_back(yield2.ptr());
	    pars.push_back(mass.ptr());
	    pars.push_back(gamma.ptr());
	    pars.push_back(mean1.ptr());
	    pars.push_back(sigma1.ptr());
	    pars.push_back(mean2.ptr());
	    pars.push_back(sigma2.ptr());
	    TF1 fun = root::tf1("fun", f, fMin, fMax, pars);
	    fun.SetParNames(yield1.name().c_str(), yield2.name().c_str(), 
			    mass.name().c_str(), gamma.name().c_str(), 
			    mean1.name().c_str(), sigma1.name().c_str(), 
			    mean2.name().c_str(), sigma2.name().c_str());
	    fun.SetLineColor(kRed);
	    TCanvas *canvas = new TCanvas("canvas");
	    zMass->Draw("e");
	    fun.Draw("same");
	    string epsFilename = "ZMassFitCoBwGGf_" + v_eps[i];
	    canvas->SaveAs(epsFilename.c_str());
	    canvas->SetLogy();
	    string epsLogFilename = "ZMassFitCoBwGGf_Log_" + v_eps[i];
	    canvas->SaveAs(epsLogFilename.c_str());
	  }
	}
      
      if (vm.count("convbwf2g"))
	{ 
	  cout << "Fitting histograms in input files to the convolution between the fixed Z Breit-Wigner and a linear combination of Gaussians\n";
	  cout << ">>> set pars: " << endl;
	  cout << yield1 << " ; " << dyield1 << endl; 
	  cout << yield2 << " ; " << dyield2 << endl; 
	  cout << mass << " ; " << dmass << endl; 
	  cout << gamma << " ; " << dgamma << endl; 
	  cout << mean1 << " ; " << dmean1 << endl; 
	  cout << sigma1 << " ; " << dsigma1 << endl; 
	  cout << mean2 << " ; " << dmean2 << endl; 
	  cout << sigma2 << " ; " << dsigma2 << endl;
	  for(size_t i = 0; i < v_ZMassHistos.size(); ++i) { 
	    TH1D * zMass = v_ZMassHistos[i]; 
	    Constant c1(yield1);
	    Constant c2(yield2);
	    BreitWigner bw(mass, gamma);
	    Gaussian gaus1(mean1, sigma1);
	    Gaussian gaus2(mean2, sigma2);
	    typedef Product<Constant, Gaussian> GaussProduct;
	    GaussProduct gp1 = c1 * gaus1;
	    GaussProduct gp2 = c2 * gaus2;
	    typedef Sum<GaussProduct, GaussProduct> GaussComb;
	    GaussComb gc = gp1 + gp2;
	    typedef Convolution<BreitWigner, GaussComb> FitFunction;
	    double range = 3 * max(sigma1.value(), sigma2.value());
	    FitFunction f(bw, gc, -range , range, 200);
	    cout << "set functions" << endl;
	    typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	    ChiSquared chi2(f, zMass, fMin, fMax);
	    int fullBins = chi2.degreesOfFreedom();
	    cout << "N. deg. of freedom: " << fullBins << endl;
	    fit::RootMinuit<ChiSquared> minuit(8, chi2, true);
	    minuit.setParameter(0, yield1, 10, 100, 100000);
	    //minuit.fixParameter(0);
	    minuit.setParameter(1, yield2, 10, 100, 100000);
	    //minuit.fixParameter(1);
	    minuit.setParameter(2, mass, .1, 70., 110);
	    minuit.fixParameter(2);
	    minuit.setParameter(3, gamma, 1, 1, 10);
	    minuit.fixParameter(3);
	    minuit.setParameter(4, mean1, 0.001, -0.5, 0.5);
	    //minuit.fixParameter(4);
	    minuit.setParameter(5, sigma1, 0.1, -5., 5.);
	    //minuit.fixParameter(5);
	    minuit.setParameter(6, mean2, 0.001, -10., 10.);
	    //minuit.fixParameter(6);
	    minuit.setParameter(7, sigma2, 0.1, -10., 10.); 
	    //minuit.fixParameter(7);
	    double amin = minuit.minimize();
	    cout << "fullBins = " << fullBins 
		 << "; free pars = " << minuit.getNumberOfFreeParameters() 
		 << endl;
	    unsigned int ndof = fullBins - minuit.getNumberOfFreeParameters();
	    cout << "Chi^2 = " << amin << "/" << ndof << " = " << amin/ndof 
		 << "; prob: " << TMath::Prob( amin, ndof )
		 << endl;
	    dyield1 = minuit.getParameterError(0);
	    cout << yield1 << " ; " << dyield1 << endl;
	    dyield2 = minuit.getParameterError(1);
	    cout << yield2 << " ; " << dyield2 << endl;
	    //dmass = minuit.getParameterError(2);
	    cout << mass << " ; " << dmass << endl;
	    //dgamma = minuit.getParameterError(3);
	    cout << gamma << " ; " << dgamma << endl;
	    dmean1 = minuit.getParameterError(4);
	    cout << mean1 << " ; " << dmean1 << endl;
	    dsigma1 = minuit.getParameterError(5);
	    cout << sigma1 << " ; " << dsigma1 << endl;
	    dmean2 = minuit.getParameterError(6);
	    cout << mean2 << " ; " << dmean2 << endl;
	    dsigma2 = minuit.getParameterError(7);
	    cout << sigma2 << " ; " << dsigma2 << endl;
	    vector<shared_ptr<double> > pars;
	    pars.push_back(yield1.ptr());
	    pars.push_back(yield2.ptr());
	    pars.push_back(mass.ptr());
	    pars.push_back(gamma.ptr());
	    pars.push_back(mean1.ptr());
	    pars.push_back(sigma1.ptr());
	    pars.push_back(mean2.ptr());
	    pars.push_back(sigma2.ptr());
	    TF1 fun = root::tf1("fun", f, fMin, fMax, pars);
	    fun.SetParNames(yield1.name().c_str(), yield2.name().c_str(), 
			    mass.name().c_str(), gamma.name().c_str(), 
			    mean1.name().c_str(), sigma1.name().c_str(), 
			    mean2.name().c_str(), sigma2.name().c_str());
	    fun.SetLineColor(kRed);
	    TCanvas *canvas = new TCanvas("canvas");
	    zMass->Draw("e");
	    fun.Draw("same");
	    string epsFilename = "ZMassFitCoBwfGG_" + v_eps[i];
	    canvas->SaveAs(epsFilename.c_str());
	    canvas->SetLogy();
	    string epsLogFilename = "ZMassFitCoBwfGG_Log_" + v_eps[i];
	    canvas->SaveAs(epsLogFilename.c_str());
	  }
	}
      
      if (vm.count("convbwint2gf")) 
	{ 
	  cout << "Fitting histograms in input files to the convolution between the Breit-Wigner plus Z/photon interference and a linear combination of fixed Gaussians\n";
	  cout << ">>> set pars: " << endl;
	  cout << yield1 << " ; " << dyield1 << endl; 
	  cout << yield2 << " ; " << dyield2 << endl; 
	  cout << mass << " ; " << dmass << endl; 
	  cout << gamma << " ; " << dgamma << endl; 
	  cout << mean1 << " ; " << dmean1 << endl; 
	  cout << sigma1 << " ; " << dsigma1 << endl; 
	  cout << mean2 << " ; " << dmean2 << endl; 
	  cout << sigma2 << " ; " << dsigma2 << endl;
	  for(size_t i = 0; i < v_ZMassHistos.size(); ++i) { 
	    TH1D * zMass = v_ZMassHistos[i]; 
	    Constant c1(yield1);
	    Constant c2(yield2);
	    ZLineShape zls(mass, gamma, f_gamma, f_int);
	    Gaussian gaus1(mean1, sigma1);
	    Gaussian gaus2(mean2, sigma2);
	    typedef Product<Constant, Gaussian> GaussProduct;
	    GaussProduct gp1 = c1 * gaus1;
	    GaussProduct gp2 = c2 * gaus2;
	    typedef Sum<GaussProduct, GaussProduct> GaussComb;
	    GaussComb gc = gp1 + gp2;
	    typedef Convolution<ZLineShape, GaussComb> FitFunction;
	    double range = 3 * max(sigma1.value(), sigma2.value());
	    FitFunction f(zls, gc, -range , range, 200);
	    cout << "set functions" << endl;
	    typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	    ChiSquared chi2(f, zMass, fMin, fMax);
	    int fullBins = chi2.degreesOfFreedom();
	    cout << "N. deg. of freedom: " << fullBins << endl;
	    fit::RootMinuit<ChiSquared> minuit(10, chi2, true);
	    minuit.setParameter(0, yield1, 10, 100, 100000);
	    //minuit.fixParameter(0);
	    minuit.setParameter(1, yield2, 10, 100, 100000);
	    //minuit.fixParameter(1);
	    minuit.setParameter(2, mass, .1, 70., 110);
	    //minuit.fixParameter(2);
	    minuit.setParameter(3, gamma, 1, 1, 10);
	    //minuit.fixParameter(3);
	    minuit.setParameter(4, f_gamma, 0.1, -100, 1000);
	    minuit.fixParameter(4);
	    minuit.setParameter(5, f_int, .0001, -1000000, 1000000);
	    //minuit.fixParameter(5);
	    minuit.setParameter(6, mean1, 0.001, -0.5, 0.5);
	    minuit.fixParameter(6);
	    minuit.setParameter(7, sigma1, 0.1, -5., 5.);
	    minuit.fixParameter(7);
	    minuit.setParameter(8, mean2, 0.001, -5., 5.);
	    minuit.fixParameter(8);
	    minuit.setParameter(9, sigma2, 0.1, -5., 5.);
	    minuit.fixParameter(9);
	    double amin = minuit.minimize();
	    cout << "fullBins = " << fullBins 
		 << "; free pars = " << minuit.getNumberOfFreeParameters() 
		 << endl;
	    unsigned int ndof = fullBins - minuit.getNumberOfFreeParameters();
	    cout << "Chi^2 = " << amin << "/" << ndof << " = " << amin/ndof 
		 << "; prob: " << TMath::Prob( amin, ndof )
		 << endl;
	    dyield1 = minuit.getParameterError(0);
	    cout << yield1 << " ; " << dyield1 << endl;
	    dyield2 = minuit.getParameterError(1);
	    cout << yield2 << " ; " << dyield2 << endl;
	    dmass = minuit.getParameterError(2);
	    cout << mass << " ; " << dmass << endl;
	    dgamma = minuit.getParameterError(3);
	    cout << gamma << " ; " << dgamma << endl;
	    //df_gamma = minuit.getParameterError(4);
	    cout << f_gamma << " ; " << df_gamma << endl;
	    df_int = minuit.getParameterError(5);
	    cout << f_int << " ; " << df_int << endl;
	    //dmean1 = minuit.getParameterError(6);
	    cout << mean1 << " ; " << dmean1 << endl;
	    //dsigma1 = minuit.getParameterError(7);
	    cout << sigma1 << " ; " << dsigma1 << endl;
	    //dmean2 = minuit.getParameterError(8);
	    cout << mean2 << " ; " << dmean2 << endl;
	    //dsigma2 = minuit.getParameterError(9);
	    cout << sigma2 << " ; " << dsigma2 << endl;
	    vector<shared_ptr<double> > pars;
	    pars.push_back(yield1.ptr());
	    pars.push_back(yield2.ptr());
	    pars.push_back(mass.ptr());
	    pars.push_back(gamma.ptr());
	    pars.push_back(f_gamma.ptr());
	    pars.push_back(f_int.ptr());
	    pars.push_back(mean1.ptr());
	    pars.push_back(sigma1.ptr());
	    pars.push_back(mean2.ptr());
	    pars.push_back(sigma2.ptr());
	    TF1 fun = root::tf1("fun", f, fMin, fMax, pars);
	    fun.SetParNames(yield1.name().c_str(), yield2.name().c_str(), 
			    mass.name().c_str(), gamma.name().c_str(), 
			    f_gamma.name().c_str(), f_int.name().c_str(), 
			    mean1.name().c_str(), sigma1.name().c_str(), 
			    mean2.name().c_str(), sigma2.name().c_str());
	    fun.SetLineColor(kRed);
	    TCanvas *canvas = new TCanvas("canvas");
	    zMass->Draw("e");
	    fun.Draw("same");
	    string epsFilename = "ZMassFitCoBwInGGf_" + v_eps[i];
	    canvas->SaveAs(epsFilename.c_str());
	    canvas->SetLogy();
	    string epsLogFilename = "ZMassFitCoBwInGGf_Log_" + v_eps[i];
	    canvas->SaveAs(epsLogFilename.c_str());
	  }
	}

      if(vm.count("convbwintf2g"))
	{ 
	  cout << "Fitting histograms in input files to the convolution between the fixed Breit-Wigner plus Z/photon interference and a linear combination of Gaussians\n";
	  cout << ">>> set pars: " << endl;
	  cout << yield1 << " ; " << dyield1 << endl; 
	  cout << yield2 << " ; " << dyield2 << endl; 
	  cout << mass << " ; " << dmass << endl; 
	  cout << gamma << " ; " << dgamma << endl; 
	  cout << mean1 << " ; " << dmean1 << endl; 
	  cout << sigma1 << " ; " << dsigma1 << endl; 
	  cout << mean2 << " ; " << dmean2 << endl; 
	  cout << sigma2 << " ; " << dsigma2 << endl;
	  for(size_t i = 0; i < v_ZMassHistos.size(); ++i) { 
	    TH1D * zMass = v_ZMassHistos[i]; 
	    Constant c1(yield1);
	    Constant c2(yield2);
	    ZLineShape zls(mass, gamma, f_gamma, f_int);
	    Gaussian gaus1(mean1, sigma1);
	    Gaussian gaus2(mean2, sigma2);
	    typedef Product<Constant, Gaussian> GaussProduct;
	    GaussProduct gp1 = c1 * gaus1;
	    GaussProduct gp2 = c2 * gaus2;
	    typedef Sum<GaussProduct, GaussProduct> GaussComb;
	    GaussComb gc = gp1 + gp2;
	    typedef Convolution<ZLineShape, GaussComb> FitFunction;
	    double range = 3 * max(sigma1.value(), sigma2.value());
	    FitFunction f(zls, gc, -range , range, 200);
	    cout << "set functions" << endl;
	    typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	    ChiSquared chi2(f, zMass, fMin, fMax);
	    int fullBins = chi2.degreesOfFreedom();
	    cout << "N. deg. of freedom: " << fullBins << endl;
	    fit::RootMinuit<ChiSquared> minuit(10, chi2, true);
	    minuit.setParameter(0, yield1, 10, 100, 100000);
	    //minuit.fixParameter(0);
	    minuit.setParameter(1, yield2, 10, 100, 100000);
	    //minuit.fixParameter(1);
	    minuit.setParameter(2, mass, .1, 70., 110);
	    minuit.fixParameter(2);
	    minuit.setParameter(3, gamma, 1, 1, 10);
	    minuit.fixParameter(3);
	    minuit.setParameter(4, f_gamma, 0.1, -100, 1000);
	    minuit.fixParameter(4);
	    minuit.setParameter(5, f_int, .0001, -1000000, 1000000);
	    //minuit.fixParameter(5);
	    minuit.setParameter(6, mean1, 0.001, -0.5, 0.5);
	    //minuit.fixParameter(6);
	    minuit.setParameter(7, sigma1, 0.1, -5., 5.);
	    //minuit.fixParameter(7);
	    minuit.setParameter(8, mean2, 0.001, -10., 10.);
	    //minuit.fixParameter(8);
	    minuit.setParameter(9, sigma2, 0.1, -10., 10.);
	    //minuit.fixParameter(9);
	    double amin = minuit.minimize();
	    cout << "fullBins = " << fullBins 
		 << "; free pars = " << minuit.getNumberOfFreeParameters() 
		 << endl;
	    unsigned int ndof = fullBins - minuit.getNumberOfFreeParameters();
	    cout << "Chi^2 = " << amin << "/" << ndof << " = " << amin/ndof 
		 << "; prob: " << TMath::Prob( amin, ndof )
		 << endl;
	    dyield1 = minuit.getParameterError(0);
	    cout << yield1 << " ; " << dyield1 << endl;
	    dyield2 = minuit.getParameterError(1);
	    cout << yield2 << " ; " << dyield2 << endl;
	    //dmass = minuit.getParameterError(2);
	    cout << mass << " ; " << dmass << endl;
	    //dgamma = minuit.getParameterError(3);
	    cout << gamma << " ; " << dgamma << endl;
	    //df_gamma = minuit.getParameterError(4);
	    cout << f_gamma << " ; " << df_gamma << endl;
	    df_int = minuit.getParameterError(5);
	    cout << f_int << " ; " << df_int << endl;
	    dmean1 = minuit.getParameterError(6);
	    cout << mean1 << " ; " << dmean1 << endl;
	    dsigma1 = minuit.getParameterError(7);
	    cout << sigma1 << " ; " << dsigma1 << endl;
	    dmean2 = minuit.getParameterError(8);
	    cout << mean2 << " ; " << dmean2 << endl;
	    dsigma2 = minuit.getParameterError(9);
	    cout << sigma2 << " ; " << dsigma2 << endl;
	    vector<shared_ptr<double> > pars;
	    pars.push_back(yield1.ptr());
	    pars.push_back(yield2.ptr());
	    pars.push_back(mass.ptr());
	    pars.push_back(gamma.ptr());
	    pars.push_back(f_gamma.ptr());
	    pars.push_back(f_int.ptr());
	    pars.push_back(mean1.ptr());
	    pars.push_back(sigma1.ptr());
	    pars.push_back(mean2.ptr());
	    pars.push_back(sigma2.ptr());
	    TF1 fun = root::tf1("fun", f, fMin, fMax, pars);
	    fun.SetParNames(yield1.name().c_str(), yield2.name().c_str(), 
			    mass.name().c_str(), gamma.name().c_str(), 
			    f_gamma.name().c_str(), f_int.name().c_str(), 
			    mean1.name().c_str(), sigma1.name().c_str(), 
			    mean2.name().c_str(), sigma2.name().c_str());
	    fun.SetLineColor(kRed);
	    TCanvas *canvas = new TCanvas("canvas");
	    zMass->Draw("e");
	    fun.Draw("same");
	    string epsFilename = "ZMassFitCoBwInfGG_" + v_eps[i];
	    canvas->SaveAs(epsFilename.c_str());
	    canvas->SetLogy();
	    string epsLogFilename = "ZMassFitCoBwInfGG_Log_" + v_eps[i];
	    canvas->SaveAs(epsLogFilename.c_str());
	  }
	}
      
      if (vm.count("convbwintgam2gf")) 
	{ 
	  cout << "Fitting histograms in input files to the convolution of the Breit-Wigner plus Z/photon interference and photon propagator with a linear combination of fixed Gaussians\n";
	  cout << ">>> set pars: " << endl;
	  cout << yield1 << " ; " << dyield1 << endl; 
	  cout << yield2 << " ; " << dyield2 << endl; 
	  cout << mass << " ; " << dmass << endl; 
	  cout << gamma << " ; " << dgamma << endl; 
	  cout << mean1 << " ; " << dmean1 << endl; 
	  cout << sigma1 << " ; " << dsigma1 << endl; 
	  cout << mean2 << " ; " << dmean2 << endl; 
	  cout << sigma2 << " ; " << dsigma2 << endl;
	  for(size_t i = 0; i < v_ZMassHistos.size(); ++i) { 
	    TH1D * zMass = v_ZMassHistos[i]; 
	    Constant c1(yield1);
	    Constant c2(yield2);
	    ZLineShape zls(mass, gamma, f_gamma, f_int);
	    Gaussian gaus1(mean1, sigma1);
	    Gaussian gaus2(mean2, sigma2);
	    typedef Product<Constant, Gaussian> GaussProduct;
	    GaussProduct gp1 = c1 * gaus1;
	    GaussProduct gp2 = c2 * gaus2;
	    typedef Sum<GaussProduct, GaussProduct> GaussComb;
	    GaussComb gc = gp1 + gp2;
	    typedef Convolution<ZLineShape, GaussComb> FitFunction;
	    double range = 3 * max(sigma1.value(), sigma2.value());
	    FitFunction f(zls, gc, -range , range, 200);
	    cout << "set functions" << endl;
	    typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	    ChiSquared chi2(f, zMass, fMin, fMax);
	    int fullBins = chi2.degreesOfFreedom();
	    cout << "N. deg. of freedom: " << fullBins << endl;
	    fit::RootMinuit<ChiSquared> minuit(10, chi2, true);
	    minuit.setParameter(0, yield1, 10, 100, 100000);
	    //minuit.fixParameter(0);
	    minuit.setParameter(1, yield2, 10, 100, 100000);
	    //minuit.fixParameter(1);
	    minuit.setParameter(2, mass, .1, 70., 110);
	    //minuit.fixParameter(2);
	    minuit.setParameter(3, gamma, 1, 1, 10);
	    //minuit.fixParameter(3);
	    minuit.setParameter(4, f_gamma, 0.1, -100, 1000);
	    //minuit.fixParameter(4);
	    minuit.setParameter(5, f_int, .0001, -1000000, 1000000);
	    //minuit.fixParameter(5);
	    minuit.setParameter(6, mean1, 0.001, -0.5, 0.5);
	    minuit.fixParameter(6);
	    minuit.setParameter(7, sigma1, 0.1, -5., 5.);
	    minuit.fixParameter(7);
	    minuit.setParameter(8, mean2, 0.001, -5., 5.);
	    minuit.fixParameter(8);
	    minuit.setParameter(9, sigma2, 0.1, -5., 5.);
	    minuit.fixParameter(9);
	    double amin = minuit.minimize();
	    cout << "fullBins = " << fullBins 
		 << "; free pars = " << minuit.getNumberOfFreeParameters() 
		 << endl;
	    unsigned int ndof = fullBins - minuit.getNumberOfFreeParameters();
	    cout << "Chi^2 = " << amin << "/" << ndof << " = " << amin/ndof 
		 << "; prob: " << TMath::Prob( amin, ndof )
		 << endl;
	    dyield1 = minuit.getParameterError(0);
	    cout << yield1 << " ; " << dyield1 << endl;
	    dyield2 = minuit.getParameterError(1);
	    cout << yield2 << " ; " << dyield2 << endl;
	    dmass = minuit.getParameterError(2);
	    cout << mass << " ; " << dmass << endl;
	    dgamma = minuit.getParameterError(3);
	    cout << gamma << " ; " << dgamma << endl;
	    df_gamma = minuit.getParameterError(4);
	    cout << f_gamma << " ; " << df_gamma << endl;
	    df_int = minuit.getParameterError(5);
	    cout << f_int << " ; " << df_int << endl;
	    //dmean1 = minuit.getParameterError(6);
	    cout << mean1 << " ; " << dmean1 << endl;
	    //dsigma1 = minuit.getParameterError(7);
	    cout << sigma1 << " ; " << dsigma1 << endl;
	    //dmean2 = minuit.getParameterError(8);
	    cout << mean2 << " ; " << dmean2 << endl;
	    //dsigma2 = minuit.getParameterError(9);
	    cout << sigma2 << " ; " << dsigma2 << endl;
	    vector<shared_ptr<double> > pars;
	    pars.push_back(yield1.ptr());
	    pars.push_back(yield2.ptr());
	    pars.push_back(mass.ptr());
	    pars.push_back(gamma.ptr());
	    pars.push_back(f_gamma.ptr());
	    pars.push_back(f_int.ptr());
	    pars.push_back(mean1.ptr());
	    pars.push_back(sigma1.ptr());
	    pars.push_back(mean2.ptr());
	    pars.push_back(sigma2.ptr());
	    TF1 fun = root::tf1("fun", f, fMin, fMax, pars);
	    fun.SetParNames(yield1.name().c_str(), yield2.name().c_str(), 
			    mass.name().c_str(), gamma.name().c_str(), 
			    f_gamma.name().c_str(), f_int.name().c_str(), 
			    mean1.name().c_str(), sigma1.name().c_str(), 
			    mean2.name().c_str(), sigma2.name().c_str());
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
      
      if(vm.count("convbwintgamf2g"))
	{ 
	  cout << "Fitting histograms in input files to the convolution of the fixed Breit-Wigner plus Z/photon interference and photon propagator with a linear combination of Gaussians\n";
	  cout << ">>> set pars: " << endl;
	  cout << yield1 << " ; " << dyield1 << endl; 
	  cout << yield2 << " ; " << dyield2 << endl; 
	  cout << mass << " ; " << dmass << endl; 
	  cout << gamma << " ; " << dgamma << endl; 
	  cout << mean1 << " ; " << dmean1 << endl; 
	  cout << sigma1 << " ; " << dsigma1 << endl; 
	  cout << mean2 << " ; " << dmean2 << endl; 
	  cout << sigma2 << " ; " << dsigma2 << endl;
	  for(size_t i = 0; i < v_ZMassHistos.size(); ++i) { 
	    TH1D * zMass = v_ZMassHistos[i]; 
	    Constant c1(yield1);
	    Constant c2(yield2);
	    ZLineShape zls(mass, gamma, f_gamma, f_int);
	    Gaussian gaus1(mean1, sigma1);
	    Gaussian gaus2(mean2, sigma2);
	    typedef Product<Constant, Gaussian> GaussProduct;
	    GaussProduct gp1 = c1 * gaus1;
	    GaussProduct gp2 = c2 * gaus2;
	    typedef Sum<GaussProduct, GaussProduct> GaussComb;
	    GaussComb gc = gp1 + gp2;
	    typedef Convolution<ZLineShape, GaussComb> FitFunction;
	    double range = 3 * max(sigma1.value(), sigma2.value());
	    FitFunction f(zls, gc, -range , range, 200);
	    cout << "set functions" << endl;
	    typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	    ChiSquared chi2(f, zMass, fMin, fMax);
	    int fullBins = chi2.degreesOfFreedom();
	    cout << "N. deg. of freedom: " << fullBins << endl;
	    fit::RootMinuit<ChiSquared> minuit(10, chi2, true);
	    minuit.setParameter(0, yield1, 10, 100, 100000);
	    //minuit.fixParameter(0);
	    minuit.setParameter(1, yield2, 10, 100, 100000);
	    //minuit.fixParameter(1);
	    minuit.setParameter(2, mass, .1, 70., 110);
	    minuit.fixParameter(2);
	    minuit.setParameter(3, gamma, 1, 1, 10);
	    minuit.fixParameter(3);
	    minuit.setParameter(4, f_gamma, 0.1, -100, 1000);
	    //minuit.fixParameter(4);
	    minuit.setParameter(5, f_int, .0001, -1000000, 1000000);
	    //minuit.fixParameter(5);
	    minuit.setParameter(6, mean1, 0.001, -0.5, 0.5);
	    //minuit.fixParameter(6);
	    minuit.setParameter(7, sigma1, 0.1, -5., 5.);
	    //minuit.fixParameter(7);
	    minuit.setParameter(8, mean2, 0.001, -10., 10.);
	    //minuit.fixParameter(8);
	    minuit.setParameter(9, sigma2, 0.1, -10., 10.);
	    //minuit.fixParameter(9);
	    double amin = minuit.minimize();
	    cout << "fullBins = " << fullBins 
		 << "; free pars = " << minuit.getNumberOfFreeParameters() 
		 << endl;
	    unsigned int ndof = fullBins - minuit.getNumberOfFreeParameters();
	    cout << "Chi^2 = " << amin << "/" << ndof << " = " << amin/ndof 
		 << "; prob: " << TMath::Prob( amin, ndof )
		 << endl;
	    dyield1 = minuit.getParameterError(0);
	    cout << yield1 << " ; " << dyield1 << endl;
	    dyield2 = minuit.getParameterError(1);
	    cout << yield2 << " ; " << dyield2 << endl;
	    //dmass = minuit.getParameterError(2);
	    cout << mass << " ; " << dmass << endl;
	    //dgamma = minuit.getParameterError(3);
	    cout << gamma << " ; " << dgamma << endl;
	    df_gamma = minuit.getParameterError(4);
	    cout << f_gamma << " ; " << df_gamma << endl;
	    df_int = minuit.getParameterError(5);
	    cout << f_int << " ; " << df_int << endl;
	    dmean1 = minuit.getParameterError(6);
	    cout << mean1 << " ; " << dmean1 << endl;
	    dsigma1 = minuit.getParameterError(7);
	    cout << sigma1 << " ; " << dsigma1 << endl;
	    dmean2 = minuit.getParameterError(8);
	    cout << mean2 << " ; " << dmean2 << endl;
	    dsigma2 = minuit.getParameterError(9);
	    cout << sigma2 << " ; " << dsigma2 << endl;
	    vector<shared_ptr<double> > pars;
	    pars.push_back(yield1.ptr());
	    pars.push_back(yield2.ptr());
	    pars.push_back(mass.ptr());
	    pars.push_back(gamma.ptr());
	    pars.push_back(f_gamma.ptr());
	    pars.push_back(f_int.ptr());
	    pars.push_back(mean1.ptr());
	    pars.push_back(sigma1.ptr());
	    pars.push_back(mean2.ptr());
	    pars.push_back(sigma2.ptr());
	    TF1 fun = root::tf1("fun", f, fMin, fMax, pars);
	    fun.SetParNames(yield1.name().c_str(), yield2.name().c_str(), 
			    mass.name().c_str(), gamma.name().c_str(), 
			    f_gamma.name().c_str(), f_int.name().c_str(), 
			    mean1.name().c_str(), sigma1.name().c_str(), 
			    mean2.name().c_str(), sigma2.name().c_str());
	    fun.SetLineColor(kRed);
	    TCanvas *canvas = new TCanvas("canvas");
	    zMass->Draw("e");
	    fun.Draw("same");
	    string epsFilename = "ZMassFitCoBwInGafGG_" + v_eps[i];
	    canvas->SaveAs(epsFilename.c_str());
	    canvas->SetLogy();
	    string epsLogFilename = "ZMassFitCoBwInGafGG_Log_" + v_eps[i];
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
