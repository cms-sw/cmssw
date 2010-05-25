#include "TROOT.h"
#include "TStyle.h"
#include "TFile.h"
#include "TH1.h"
#include "TLegend.h"
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
    string numDATA;
    string denDATA;
    string numMC;
    string denMC;

    string ext;
    string pname;

    po::options_description desc("Allowed options");
    desc.add_options()
      ("help,h", "produce help message")
      ("input-file,i", po::value<string >(&file), "input file")
      ("numDATA,n", po::value<string > (&numDATA), "numHistoDATA")
      ("denDATA,d", po::value<string > (&denDATA), "denHistoDATA")
      ("numMC,m", po::value<string > (&numMC), "MCnumHisto")
      ("denMC,e", po::value<string > (&denMC), "MCdenHisto")
      ("plotname,o", po::value<string > (&pname), "plot name")
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
	string  dirDenDATA =   denDATA;
	string  dirNumDATA =   numDATA;
	string  dirDenMC =   denMC;
	string  dirNumMC =   numMC;
	
	
	TH1D * denhDATA = (TH1D*) root_file->Get( dirDenDATA.c_str()  );
	TH1D * numhDATA = (TH1D*) root_file->Get( dirNumDATA.c_str() );
 
	TH1D * denhMC = (TH1D*) root_file->Get( dirDenMC.c_str()  );
	TH1D * numhMC = (TH1D*) root_file->Get( dirNumMC.c_str() );
	
	
	const int bins = denhDATA->GetXaxis()->GetNbins();
	const double xMax = denhDATA->GetXaxis()->GetXmax();
	const double xMin = denhDATA->GetXaxis()->GetXmin();
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
	
        //data
	TH1D histo("histo", "Efficiency", bins, xMin, xMax);
	for(int i = 0; i < bins; ++i) {
	  x[i] = (double(i - 0.5 )) * (xMax - xMin) / (bins ) + xMin; 
	    int n0 = denhDATA->GetBinContent(i);
	  //	  std::cout << " n0 " << n0 << endl;
	   int n1 = numhDATA->GetBinContent(i);
	  // std::cout << " n1 " << n1 << endl;
	  if ( n0!=0) {
	    eff[i] = double(n1)/double(n0); 
	    histo.SetBinContent(i,eff[i]); 
	    exl[i] = exh[i] = 0;
	    cp.calculate(n1, n0);
	    eefflCP[i] = eff[i] - cp.lower();
	    eeffhCP[i] = cp.upper() - eff[i];
	  } else { 
	    eff[i]=0;
	    histo.SetBinContent(i,eff[i]); 
	    exl[i] = exh[i] = 0;
	    //cp.calculate(n1, n0);
	    eefflCP[i] = 0;
	    eeffhCP[i] = 0;
	    
	  }
	  //histo.SetBinContent(i+1,eff[i]); 
	      //exl[i] = exh[i] = 0;
	      //cp.calculate(n1, n0);
	      //eefflCP[i] = eff[i] - cp.lower();
	      //eeffhCP[i] = cp.upper() - eff[i];
	}
	TGraphAsymmErrors graphCP(bins, x, eff, exl, exh, eefflCP, eeffhCP);
	graphCP.SetTitle("HLT_Mu9 efficiency (Clopper-Pearson intervals)");
	graphCP.SetMarkerColor(kRed);
	graphCP.SetMarkerStyle(21);
	graphCP.SetLineWidth(2);
	graphCP.SetLineColor(kRed);
	string cname = pname;
	TCanvas * c = new TCanvas(cname.c_str());
	gStyle->SetOptStat(0);
	histo.SetTitle("HLT_Mu9 Efficiency with Clopper-Pearson intervals"); 
	histo.Draw();
	histo.SetLineColor(kWhite);
	graphCP.Draw("P");

	//mc
	double * xMC = new double[bins];
	double *effMC = new double[bins];
	double * exlMC= new double[bins];
	double *exhMC = new double[bins];
	double *  eefflCPMC= new double[bins];
	double * eeffhCPMC = new double[bins];


	TH1D histoMC("histoMC", "EfficiencyMC", bins, xMin, xMax);
	for(int i = 0; i < bins; ++i) {
	  xMC[i] = (double(i - 0.5 )) * (xMax - xMin) / (bins ) + xMin; 
	    int n0 = denhMC->GetBinContent(i);
	  //	  std::cout << " n0 " << n0 << endl;
	   int n1 = numhMC->GetBinContent(i);
	  // std::cout << " n1 " << n1 << endl;
	  if ( n0!=0) {
	    effMC[i] = double(n1)/double(n0); 
	    histoMC.SetBinContent(i,effMC[i]); 
	    exlMC[i] = exhMC[i] = 0;
	    cp.calculate(n1, n0);
	    eefflCPMC[i] = effMC[i] - cp.lower();
	    eeffhCPMC[i] = cp.upper() - effMC[i];
	  } else { 
	    effMC[i]=0;
	    histoMC.SetBinContent(i,effMC[i]); 
	    exlMC[i] = exhMC[i] = 0;
	    //cp.calculate(n1, n0);
	    eefflCPMC[i] = 0;
	    eeffhCPMC[i] = 0;
	    
	  }
	  //histo.SetBinContent(i+1,eff[i]); 
	      //exl[i] = exh[i] = 0;
	      //cp.calculate(n1, n0);
	      //eefflCP[i] = eff[i] - cp.lower();
	      //eeffhCP[i] = cp.upper() - eff[i];
	}
	TGraphAsymmErrors graphCPMC(bins, xMC, effMC, exlMC, exhMC, eefflCPMC, eeffhCPMC);
	graphCPMC.SetTitle("HLT_Mu9 efficiency (Clopper-Pearson intervals)");
	graphCPMC.SetMarkerColor(kBlue);
	graphCPMC.SetMarkerStyle(21);
	graphCPMC.SetLineWidth(2);
	graphCPMC.SetLineColor(kBlue);

	histoMC.SetTitle("HLT_Mu9 Efficiency with Clopper-Pearson intervals"); 
	histoMC.Draw("same");
	histoMC.SetLineColor(kBlue);
	histoMC.SetLineWidth(2);
	//graphCPMC.Draw("P");

	TLegend * leg = new TLegend(0.65,0.60,0.85,0.75);
	leg->SetFillColor(kWhite);
         
	leg->AddEntry(&graphCP,"data", "l");
	leg->AddEntry(&graphCPMC,"MC", "l");
	leg->Draw();



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

