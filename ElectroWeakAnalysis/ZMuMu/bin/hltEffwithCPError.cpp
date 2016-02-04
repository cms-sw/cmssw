#include "TROOT.h"
#include "TStyle.h"
#include "TFile.h"
#include "TH1.h"
#include "TGraphAsymmErrors.h" 
#include "TCanvas.h"
#include <cmath>
#include <fstream>
#include <iostream>

#include <exception>
#include <iterator>
#include <string>
#include <vector>

#if (defined (STANDALONE) or defined (__CINT__) )
    #include "ClopperPearsonBinomialInterval.h"
#else
    #include "PhysicsTools/RooStatsCms/interface/ClopperPearsonBinomialInterval.h"
#endif

#include <boost/program_options.hpp>
using namespace boost;
namespace po = boost::program_options;
using namespace std;

// A helper function to simplify the main part.
template<class T>
ostream& operator<<(ostream& os, const vector<T>& v) {
  copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
  return os;
}


int main(int ac, char *av[]) {
  gROOT->SetBatch(kTRUE);
  gROOT->SetStyle("Plain");
  
  try{
    string file;
    string num;
    string den;
    string ext;
    string pname;
    unsigned int rebin=1;
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help,h", "produce help message")
      ("input-file,i", po::value<string >(&file), "input file")
      ("num,n", po::value<string > (&num), "numHisto")
      ("den,d", po::value<string > (&den), "denHisto")
      ("plotname,o", po::value<string > (&pname), "plot name")
      ("rebin,r", po::value<unsigned int > (&rebin)->default_value(1), "rebin")
      ("plot-format,p", po::value<string>(&ext)->default_value("gif"), 
       "output plot format");
    
    po::positional_options_description p;
    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).
	      options(desc).positional(p).run(), vm); 
    po::notify(vm);



       
    if (vm.count("help")) {
      cout << "Usage: options_description [options]\n";
      cout << desc;
      return 0;
    }



    
   
   	TFile * root_file = new TFile(file.c_str(),"read");
	//TFile * root_file = new TFile("MuTrigger_133874_133828_768ub.root","read");
	//string d_string = *itd;
	string  dirDen =   den;
	string  dirNum =   num;
	
	
	TH1D * denh = (TH1D*) root_file->Get( dirDen.c_str()  );
	TH1D * numh = (TH1D*) root_file->Get( dirNum.c_str() );
	
	
         int bins = denh->GetXaxis()->GetNbins();
        bins = bins/rebin;
	const double xMax = denh->GetXaxis()->GetXmax();
  	const double xMin = denh->GetXaxis()->GetXmin();
	//std::cout << "xMax " << xMax << endl;
	//std::cout << "xMin " << xMax << endl;
	//std::cout << "bins " << bins << endl;
	double * x = new double[bins];
	double *eff = new double[bins];
	double * exl= new double[bins];
	double *exh = new double[bins];
	double *  eefflCP= new double[bins];
	double * eeffhCP = new double[bins];
	
	ClopperPearsonBinomialInterval cp;
	//  alpha = 1 - CL
	const double alpha = (1-0.682);
	cp.init(alpha);
	TH1D histo("histo", "Efficiency", bins, xMin, xMax);
	
	for(int i = 1; i <= bins; i++) {
          int j = i-1;    
	  x[j] = ((double(i-0.5)) * (xMax - xMin) / (bins )) + xMin; 
	    int n0 = denh->GetBinContent(i);
	  //	  std::cout << " n0 " << n0 << endl;
	   int n1 = numh->GetBinContent(i);
	  // std::cout << " n1 " << n1 << endl;
	  if ( n0!=0) {
	    eff[j] = double(n1)/double(n0); 
	    histo.SetBinContent(i,eff[j]); 
	    exl[j] = exh[j] = 0;
	    cp.calculate(n1, n0);
	    eefflCP[j] = eff[j] - cp.lower();
	    eeffhCP[j] = cp.upper() - eff[j];
	  } else { 
	    eff[j]=0;
	    histo.SetBinContent(i,eff[j]); 
	    exl[j] = exh[j] = 0;
	    //cp.calculate(n1, n0);
	    eefflCP[i] = 0;
	    eeffhCP[i] = 0;
	    
	  }
	  //std::cout<< "x[j] " <<x[j]<<std::endl;
	  //std::cout<< "n0 " <<n0<<std::endl;
	  //std::cout<< "n1 " <<n1<<std::endl;
	  //histo.SetBinContent(i+1,eff[i]); 
	      //exl[i] = exh[i] = 0;
	      //cp.calculate(n1, n0);
	      //eefflCP[i] = eff[i] - cp.lower();
	      //eeffhCP[i] = cp.upper() - eff[i];
	}
	TGraphAsymmErrors graphCP(bins, x, eff, exl, exh, eefflCP, eeffhCP);
	graphCP.SetTitle("trigger (HLT_Mu9 path) efficiency");
	graphCP.SetMarkerColor(kRed);
	graphCP.SetMarkerStyle(21);
	graphCP.SetLineWidth(1);
	graphCP.SetLineColor(kRed);
        string cname = pname;
	TCanvas * c = new TCanvas(cname.c_str());
	gStyle->SetOptStat(0);
	histo.SetTitle("MC trigger (HLT_Mu9 path) efficiency"); 
	histo.Draw();
	histo.SetLineColor(kWhite);
        histo.SetMinimum(0.0)
;	histo.GetXaxis()->SetTitle("p_{T} (GeV/c)");
	histo.GetYaxis()->SetTitle("efficiency");
        graphCP.Draw("P");
    	string plot= "HLTMu_9" +  pname + "." +  ext ; 
	c->SaveAs(plot.c_str());
        string outfile= "Effhisto" +   pname + ".root";
	TFile * output_file = TFile::Open(outfile.c_str(), "recreate");
        string outdir =  pname;
	TDirectory * dir = output_file->mkdir(outdir.c_str());
	dir->cd();
        c->Write();  
        histo.Write();
        graphCP.Write();  
	output_file->Close();
        delete c;
      }    
   
  
  catch(std::exception& e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
  }
  catch(...) {
    cerr << "Exception of unknown type!\n";
  }

  return 0;
}

