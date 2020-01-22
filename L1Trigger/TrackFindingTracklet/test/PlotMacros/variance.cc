#include "TMath.h"
#include "TRint.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TLorentzVector.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TGaxis.h"
#include <fstream>
#include <iostream>
#include "TMath.h"
#include <string>
#include <map>

class Averager{

public:
  
  Averager() {
    n_=0;
    sum_=0;
  }

  void Fill(double x){
    sum_+=x;
    n_++;
  }

  double GetMean() {
    return sum_/n_;
  }

  double GetEntries() {
    return n_;
  }

private:

  int n_;
  double sum_;
  
};


void variance(){
//
// To see the output of this macro, click here.

//

#include "TMath.h"

gROOT->Reset();

gROOT->SetStyle("Plain");

gStyle->SetCanvasColor(kWhite);

gStyle->SetCanvasBorderMode(0);     // turn off canvas borders
gStyle->SetPadBorderMode(0);
gStyle->SetOptStat(1111);
gStyle->SetOptTitle(1);

  // For publishing:
  gStyle->SetLineWidth(1);
  gStyle->SetTextSize(1.1);
  gStyle->SetLabelSize(0.06,"xy");
  gStyle->SetTitleSize(0.06,"xy");
  gStyle->SetTitleOffset(1.2,"x");
  gStyle->SetTitleOffset(1.0,"y");
  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadRightMargin(0.1);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.12);


 
 ofstream out("variance.dat");

 double ptmin=1.0;
 double ptmax=2.0;
 
 for(unsigned int ptbin=0;ptbin<4;ptbin++) {


  std::vector<std::pair<std::pair<std::string,std::string>,std::vector<std::pair<std::pair<int,int>, Averager> > > > histlist;


  ifstream fitpatternin("fitpattern.txt");

  double min=-1;
  double max=1;
  
  while (fitpatternin.good()) {

    string layer,disk;
    int n;

    fitpatternin >> layer >> disk >> n;

    if (!fitpatternin.good()) continue;

    std::pair<std::string,std::string> layerdisk(layer,disk);

    std::vector<std::pair<std::pair<int,int>, Averager> > hists;

    for(int i=0;i<11;i++){
      if (i<6) {
	if (layer[i]=='0') continue;
      } else {
	if (disk[2*(i-6)]=='0' && disk[2*(i-6)+1]=='0' ) continue;
      }
      for(int j=0;j<11;j++){
	if (j<6) {
	  if (layer[j]=='0') continue;
	} else {
	  if (disk[2*(j-6)]=='0' && disk[2*(j-6)+1]=='0' ) continue;
	}
	std::pair<int,int> indexpair(i,j);

	TString name="LayerDisk ";
	name+=layer;
	name+=" ";
	name+=disk;
	name+=" ";
	name+=i;
	name+="-";
	name+=j;
      
	
        Averager avg;

	std::pair<std::pair<int,int>, Averager> histpair(indexpair,avg);

	
	hists.push_back(histpair);
	

	
      }
    }

    std::pair<std::pair<std::string,std::string>,std::vector<std::pair<std::pair<int,int>, Averager> > >  patternentry(layerdisk,hists);
    
    histlist.push_back(patternentry);
    
  }

  fitpatternin.close();

   

   
   ptmin*=2;
   ptmax*=2;
   
   ifstream in("variance.txt");

   int count=0;

   double pt, r[11];
   
   while (in.good()){

     in >>pt>>r[0]>>r[1]>>r[2]>>r[3]>>r[4]>>r[5]>>r[6]>>r[7]>>r[8]>>r[9]>>r[10];

     bool large=false;

     for (unsigned int i=0;i<6;i++) {
       if (r[i]<-9999) continue;
       if (fabs(r[i])>0.2) large=true;
     }
     
     if (large) { 
       cout <<r[0]<<"  "<<r[1]<<"  "<<r[2]<<"  "<<r[3]<<"  "<<r[4]<<"  "
	    <<r[5]<<"  "<<r[6]<<"  "<<r[7]<<"  "<<r[8]<<"  "<<r[9]<<"  "
	    <<r[10]<<endl;
     }

     if (large) continue;
       
     if (!in.good()) continue;

     if (pt<ptmin) continue;
     if (pt>ptmax) continue;

     for(unsigned int j=0;j<histlist.size();j++){

       std::string layer=histlist[j].first.first;
       std::string disk=histlist[j].first.second;

       bool match=true;
       
       for(int ii=0;ii<6;ii++) {
	 if (layer[ii]=='1' && r[ii]<-9999) match=false;
       }
       for(int ii=0;ii<5;ii++) {
	 if ((disk[2*ii]=='1' || disk[2*ii+1]=='1') && r[ii+6]<-9999) match=false;
       }

       if (!match) continue;
       
       for(unsigned int i=0;i<histlist[j].second.size();i++) {

	 int l1=histlist[j].second[i].first.first;
	 if (r[l1]<-9999) continue;
	 
	 int l2=histlist[j].second[i].first.second;
	 if (r[l2]<-9999) continue;

	 //if (layer=="111111") {
	 //  cout <<layer<<" "<<l1<<" "<<l2<<" "<<r[l1]<<" "<<r[l2]<<endl;
	 //}
	 
	 histlist[j].second[i].second.Fill(r[l1]*r[l2]);
     
       }

     }

     count++;

   }

   cout << "Processed: "<<count<<" events"<<endl;

   TCanvas* c=0;

   for(unsigned int  j=0;j<histlist.size();j++) {
     out <<"V "<<ptbin<<" "<<histlist[j].first.first<<" "<<histlist[j].first.second<<endl;
     for(unsigned int i=0;i<histlist[j].second.size();i++) {
       if (histlist[j].second[i].second.GetEntries()>100) {
	 out <<"E "<<histlist[j].second[i].first.first<<" "
	     <<histlist[j].second[i].first.second<<" "<<histlist[j].second[i].second.GetMean()<<" "<<histlist[j].second[i].second.GetEntries()<<endl;
       }
     }
   }

 }
   
 out.close();
 
}

