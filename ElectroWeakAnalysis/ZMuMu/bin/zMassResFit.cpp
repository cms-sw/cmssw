#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "PhysicsTools/Utilities/interface/Constant.h"
#include "PhysicsTools/Utilities/interface/Product.h"
#include "PhysicsTools/Utilities/interface/Sum.h"
#include "PhysicsTools/Utilities/interface/Gaussian.h"
#include "PhysicsTools/Utilities/interface/HistoChiSquare.h"
#include "PhysicsTools/Utilities/interface/RootMinuit.h"
#include "PhysicsTools/Utilities/interface/RootFunctionAdapter.h"
#include "TFile.h"
#include "TH1.h"
#include "TF1.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TMath.h"
#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>
using namespace boost;
namespace po = boost::program_options;

#include <iostream>
#include <algorithm> 
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
	("min,m", po::value<double>(&fMin)->default_value(-20), "minimum value for fit range")
	("max,M", po::value<double>(&fMax)->default_value(20), "maximum value for fit range")
	("gauss", "fit to a gaussian")
	("2gauss", "fit to a linear combination of two gaussians")
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
      vector<TH1D*> v_ZMassResHistos;
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
	    TDirectory *RecoHistos = (TDirectory*) Histos->GetDirectory("ZResolutionHisto");
	    TH1D * zMass = (TH1D*) RecoHistos->Get("ZToMuMuRecoMassResolution");
	    zMass->GetXaxis()->SetTitle("Mass (GeV/c^{2})"); 
	    v_ZMassResHistos.push_back(zMass);
	    gROOT->SetStyle("Plain");
	    string f_string = *it;
	    replace(f_string.begin(), f_string.end(), '.', '_');
	    string eps_string = f_string + ext;
	    v_eps.push_back(eps_string);
	    cout << ">>> histogram loaded\n";
	  }
	  cout << v_file.size() << ", " << v_ZMassResHistos.size() << ", " << v_eps.size() << endl;
	  cout <<">>> Input files loaded\n";
	}
      //Parameters for fit
      Parameter yield1("Yield 1", 1000);
      Parameter yield2("Yield 2", 1000);
      Parameter mean1("Mean 1", 0);
      Parameter sigma1("Sigma 1", 1.);
      Parameter mean2("Mean 2", 0);
      Parameter sigma2("Sigma 2", 1.);
      Parameter dyield1("Yield 1 Error", 0);
      Parameter dyield2("Yield 2 Error", 0);
      Parameter dmean1("Mean 1 Error", 0);
      Parameter dsigma1("Sigma 1 Error", 0); 
      Parameter dmean2("Mean 2 Error", 0);
      Parameter dsigma2("Sigma 2 Error", 0); 
      
      if (vm.count("gauss"))
	{
	  cout << "Fitting histograms in input files to a Gaussian\n"; 
	  cout << ">>> set pars: " << endl;
	  cout << yield1 << " ; " << dyield1 << endl; 
	  cout << mean1 << " ; " << dmean1 << endl; 
	  cout << sigma1 << " ; " << dsigma1 << endl;
	  for(size_t i = 0; i < v_ZMassResHistos.size(); ++i) { 
	    TH1D * zMass = v_ZMassResHistos[i]; 
	    Gaussian gaus(mean1, sigma1);
	    Constant c(yield1);
	    typedef Product<Constant, Gaussian> FitFunction;
	    FitFunction f = c * gaus;
	    typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	    ChiSquared chi2(f, zMass, fMin, fMax);
	    int fullBins = chi2.degreesOfFreedom();
	    cout << "N. deg. of freedom: " << fullBins << endl;
	    fit::RootMinuit<ChiSquared> minuit(3, chi2, true);
	    minuit.setParameter(0, yield1, 10, 100, 100000);
	    minuit.setParameter(1, mean1, 0.001, -1., 1.);
	    minuit.fixParameter(1);
	    minuit.setParameter(2, sigma1, 0.1, -5., 5.);
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
	    //dmean1 = minuit.getParameterError(1);
	    cout << mean1 << " ; " << dmean1 << endl;
	    dsigma1 = minuit.getParameterError(2);
	    cout << sigma1 << " ; " << dsigma1 << endl;
	    TF1 fun = root::tf1("fun", f, fMin, fMax, yield1, mean1, sigma1);
	    fun.SetParNames(yield1.name().c_str(), mean1.name().c_str(), sigma1.name().c_str());
	    fun.SetLineColor(kRed);
	    TCanvas *canvas = new TCanvas("canvas");
	    zMass->Draw("e");
	    fun.Draw("same");	
	    string epsFilename = "ZMassResFitG_" + v_eps[i];
	    canvas->SaveAs(epsFilename.c_str());
	    canvas->SetLogy();
	    string epsLogFilename = "ZMassResFitG_Log_" + v_eps[i];
	    canvas->SaveAs(epsLogFilename.c_str());
	  }
	}
      
     if (vm.count("2gauss"))
	{
	  cout << "Fitting histograms in input files to a linear combination of two Gaussians\n"; 
	  cout << "set pars: " << endl;
	  cout << yield1 << " ; " << dyield1 << endl;
	  cout << yield2 << " ; " << dyield2 << endl;
	  cout << mean1 << " ; " << dmean1 << endl; 
	  cout << sigma1 << " ; " << dsigma1 << endl; 
	  cout << mean2 << " ; " << dmean2 << endl; 
	  cout << sigma2 << " ; " << dsigma2 << endl; 
	  for(size_t i = 0; i < v_ZMassResHistos.size(); ++i) { 
	    TH1D * zMass = v_ZMassResHistos[i]; 
	    Gaussian gaus1(mean1, sigma1);
	    Gaussian gaus2(mean2, sigma2);
	    Constant c1(yield1);
	    Constant c2(yield2);
	    typedef Product<Constant, Gaussian> ConstGaussian;
	    ConstGaussian cog1 = c1 * gaus1;
	    ConstGaussian cog2 = c2 * gaus2;
	    typedef Sum<ConstGaussian, ConstGaussian> FitFunction;
	    FitFunction f = cog1 + cog2;
	    typedef fit::HistoChiSquare<FitFunction> ChiSquared;
	    ChiSquared chi2(f, zMass, fMin, fMax);
	    int fullBins = chi2.degreesOfFreedom();
	    cout << "N. deg. of freedom: " << fullBins << endl;
	    fit::RootMinuit<ChiSquared> minuit(6, chi2, true);
	    minuit.setParameter(0, yield1, 10, 100, 100000);
	    minuit.setParameter(1, yield2, 10, 100, 100000);
	    minuit.setParameter(2, mean1, 0.001, -1., 1.);
	    //minuit.fixParameter(2);
	    minuit.setParameter(3, sigma1, 0.1, -5., 5.);
	    minuit.setParameter(4, mean2, 0.001, -2, 2.);
	    minuit.setParameter(5, sigma2, 0.1, -5., 5.);
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
	    dmean1 = minuit.getParameterError(2);
	    cout << mean1 << " ; " << dmean1 << endl;
	    dsigma1 = minuit.getParameterError(3);
	    cout << sigma1 << " ; " << dsigma1 << endl;
	    dmean2 = minuit.getParameterError(4);
	    cout << mean2 << " ; " << dmean2 << endl;
	    dsigma2 = minuit.getParameterError(5);
	    cout << sigma2 << " ; " << dsigma2 << endl;
	    vector<shared_ptr<double> > pars;
	    pars.push_back(yield1.ptr());
	    pars.push_back(yield2.ptr());
	    pars.push_back(mean1.ptr());
	    pars.push_back(sigma1.ptr());
	    pars.push_back(mean2.ptr());
	    pars.push_back(sigma2.ptr());
	    TF1 fun = root::tf1("fun", f, fMin, fMax, pars);
	    fun.SetParNames(yield1.name().c_str(), yield2.name().c_str(), 
			    mean1.name().c_str(), sigma1.name().c_str(), 
		  mean2.name().c_str(), sigma2.name().c_str());
	    fun.SetLineColor(kRed);
	    TCanvas *canvas = new TCanvas("canvas");
	    zMass->Draw("e");
	    fun.Draw("same");	
	    string epsFilename = "ZMassResFitGG_" + v_eps[i];
	    canvas->SaveAs(epsFilename.c_str());
	    canvas->SetLogy();
	    string epsLogFilename = "ZMassResFitGG_Log_" + v_eps[i];
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
