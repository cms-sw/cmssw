#include <vector>
#include <string>
using namespace std;

int colorList[] = {1,2,3,4,5,6,7,8,9,10}; 
int markerStyleList[] = {21,21,23,23,22,22,23,23,21,21};  

TObject * getHistogram(TFile * f, string algo,string histoName, string range = "GLOBAL")
{
string prefix = "DQMData/Btag/JetTag";
string d = prefix+"_"+algo+"_"+range;
 cout <<" DIR "<<d<<endl;
TDirectory * dir  =(TDirectory *) f->Get(d.c_str());
return dir->Get((histoName+"_"+algo+"_"+range).c_str());
}


void setStyle(int i, TH1F *obj)
{
obj->SetMarkerColor(colorList[i]);
obj->SetMarkerStyle(markerStyleList[i]);
}



void drawAll()
{
  //  TFile *_file0 = TFile::Open("jetTagAnalysisBoris_standard.root");
  TFile *_file1 = TFile::Open("DQM_BTAG.root");
  
  vector<TFile *> files;
  vector<string> algos;
  algos.push_back("trackCountingHighPurBJetTags");
  algos.push_back("trackCountingHighEffBJetTags");
  algos.push_back("jetProbabilityBJetTags");
  algos.push_back("jetBProbabilityBJetTags");
  algos.push_back("simpleSecondaryVertexHighEffBJetTags");
  algos.push_back("simpleSecondaryVertexHighPurBJetTags");
  algos.push_back("combinedSecondaryVertexBJetTags");
  algos.push_back("combinedSecondaryVertexMVABJetTags");
  algos.push_back("ghostTrackBJetTags");
  algos.push_back("softMuonBJetTags");
  algos.push_back("softMuonNoIPBJetTags");
  algos.push_back("softElectronBJetTags");

  files.push_back(_file1);
  files.push_back(_file1);
  files.push_back(_file1);
  files.push_back(_file1);
  files.push_back(_file1);
  files.push_back(_file1);
  files.push_back(_file1);
  files.push_back(_file1);
  files.push_back(_file1);
  files.push_back(_file1);
  files.push_back(_file1);
  files.push_back(_file1);


  TLegend * leg = new TLegend(0.4,0.4,0.6,0.6);
  TCanvas * c1 = new TCanvas();
  c1->SetLogy();  
  c1->SetGridy();  
  c1->SetGridx();  
  for(int i = 0 ; i < algos.size() ; i++)
   {
      cout << algos[i] << endl;
      //TH1F * h = (TH1F *) getHistogram(files[i],algos[i],"FlavEffVsBEff_DUS_discr","ETA_0-1.4");
     TH1F * h = (TH1F *) getHistogram(files[i],algos[i],"FlavEffVsBEff_DUS_discr","GLOBAL");
     //TH1F * h = (TH1F *) getHistogram(files[i],algos[i],"FlavEffVsBEff_DUS_discr","ETA_1.4-2.4");
     cout << h << endl;
     if(i==0) h->Draw(); else h->Draw("same"); 
     setStyle(i,h);
     leg->AddEntry(h,algos[i].c_str(),"p");
   }
  leg->Draw("same");

}

