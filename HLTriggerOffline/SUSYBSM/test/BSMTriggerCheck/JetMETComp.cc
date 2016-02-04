#include <TCanvas.h>
#include <TH1D.h>
#include <TText.h>
#include <TLine.h>
#include <iostream>
#include "JetMETComp.hh"

JetMETComp::JetMETComp(map<string, double> compatibilities_v){
  compatibilities =   compatibilities_v;
}

void JetMETComp::MakePlot(string name) {
  // build vector with compatibility values
  // and vector with path names
  vector<double> values;
  vector<string> names;
  map<string,double>::iterator iter;
  for( iter = compatibilities.begin(); iter != compatibilities.end(); iter++ ) {
    string tmpstring = iter->first;
    if(tmpstring.find(name) != string::npos) {
      values.push_back(iter->second);
      string tmpstring2 = iter->first;
      names.push_back(tmpstring2.erase(0, name.length()-3));
    }
  }

  string filename = name+"_compatibility";

  TH1D* comp = new TH1D(filename.c_str(), filename.c_str(), values.size(),0., double(values.size()));
  for(int i=0; i< int(values.size()); i++) {
    comp->GetXaxis()->SetBinLabel(i+1,names[i].c_str());
    comp->SetBinContent(i+1,values[i]);
  }
  
  // From JetMET code
  // create summary canvas
  TCanvas main_c("main_c","main_c",799,780);
  main_c.SetFillColor(0);
  main_c.SetBorderMode(0);

  main_c.Draw();

  TPad main_p("main_p","main_p",0.01,0.01,0.99,0.94);
  main_p.SetFillColor(0);
  main_p.SetBorderMode(0);
  main_p.SetLeftMargin(0.30);
  main_p.SetBottomMargin(0.15);
  main_p.SetLogx(1);
  main_p.SetGrid();
  main_p.SetFrameFillColor(10);
  main_p.Draw();
  
  main_c.cd();
  //   TText summary_title(.01, .95, "");
  //   summary_title.Draw("SAME");

  main_p.cd();

  // setup the passing test bars
  comp->SetStats(0);
  comp->GetXaxis()->SetLabelSize(0.04);
  //  comp->GetYaxis()->SetTitle(filename.c_str());
  comp->GetYaxis()->SetTitle("");
  comp->SetBarWidth(0.7);
  comp->SetBarOffset(0.1);
  comp->SetFillColor(38);
  comp->SetLineColor(2);
  comp->GetYaxis()->SetRangeUser(1E-7,2.);
  comp->Draw("hbar2");

//   // setup the failing test bars
//   h1dResults_failed.SetStats(0);
//   h1dResults_failed.GetXaxis()->SetLabelSize(0.04);
//   h1dResults_failed.GetYaxis()->SetTitle(filename.c_str());
//   h1dResults_failed.SetBarWidth(0.7);
//   h1dResults_failed.SetBarOffset(0.1);
//   h1dResults_failed.SetFillColor(kRed);
//   h1dResults_failed.SetLineColor(1);
//   h1dResults_failed.GetYaxis()->SetRangeUser(1E-7,2);
//   h1dResults_failed.Draw("hbar0SAME");

  // draw the pass/fail threshold line
  float threshold = std::pow(10.,-6.);
  TLine l(threshold, 0, threshold, values.size());
  l.SetLineColor(kRed);
  l.SetLineWidth(2);
  l.SetLineStyle(2);
  l.Draw("SAME"); 
  
  main_c.SaveAs(string(filename+".eps").c_str());
  
  lines.push_back("<div id=\"main_d\">");
  lines.push_back("<div id=\"main_d\">"); 
  lines.push_back("<img src=\""+filename+".jpg\" usemap=\"#"+filename+
		  "\" alt=\"\" style=\"border-style: none;\">"); 
  lines.push_back("<map id=\""+filename+"\" name=\""+filename+"\">");
//   for(int i=1; i<=names.size(); i++) {
//     char coordinates[256];
//     sprintf(coordinates,"\"241,%i,689,%i\"",88+26*(i-1),105+26*(i-1));
//     lines.push_back("<area shape=\"rect\" alt=\"\" coords="+string(coordinates)+" href=\""+
// 		    name+names[names.size()-i].erase(0,3)+".jpg\">"); 
//   }
  int barsize = (15*26/names.size());
  for(int i=1; i<=names.size(); i++) {
    char coordinates[256];
    sprintf(coordinates,"\"241,%i,689,%i\"",88+barsize*(i-1),int(88+barsize*(i-0.3)));
    lines.push_back("<area shape=\"rect\" alt=\"\" coords="+string(coordinates)+" href=\""+
		    name+names[names.size()-i].erase(0,3)+".jpg\">"); 
  }
  lines.push_back("</map>");
  lines.push_back("</div>");
}

void JetMETComp::WriteFile() {
  system("\\rm compatibility.html");
  FILE*  f=fopen("compatibility.html","w");
  for(unsigned int j = 0; j< lines.size(); j++) {
    fprintf(f,"%s\n",lines[j].c_str());
  }
  fclose(f);
}
