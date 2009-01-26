#include <memory>
#include <TH1F.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TString.h>

using namespace std;

void MakeHTML(){

  int i, j, k;
  int nhistos;

  const int NDIRS     = 19;

  const int NLIHISTOS = 363;
  const int NMUHISTOS = 283;

  const int NDRAWN_HISTOS = 13;

  string sHistName, sFileName;
  TString OutLabel, HistName;
  TString DirName;
  int DrawSwitch;

  ofstream htmls[NDIRS];
  
  ifstream DirList("DirList.txt");

  TCanvas *myc = new TCanvas("myc","",800,600);
  
  for (i = 0; i < NDIRS; i++){

    DirList>>DirName;
    cout<<DirName<<endl;
    std::auto_ptr<ifstream> HistoList;

    sFileName = DirName+".html";
    htmls[i].open(sFileName.c_str());
    sFileName.clear();

    if (i < 10){
      HistoList = std::auto_ptr<ifstream>(new ifstream("LiHistoList.txt"));
      nhistos = NLIHISTOS;
    }
    else{
      HistoList = std::auto_ptr<ifstream>(new ifstream("MuHistoList.txt"));
      nhistos = NMUHISTOS;
    }
    
    k = 0;
    for (j = 0; j < nhistos; j++){
      (*HistoList)>>sHistName>>DrawSwitch;
      if (DrawSwitch == 0) continue;
      
      if (k >= NDRAWN_HISTOS) cout<<"Histogram index is out of range"<<endl;
      
      (*HistoList)>>OutLabel;
      
      OutLabel += ("_"+DirName+".gif");
     
      if (k == 0){
	htmls[i]<<"<p>"<<endl<<"<b>"<<DirName<<"</b> <br>"<<endl;
	htmls[i]<<"<p>"<<endl<<"<font color='#aa0000'> >>> click on thumbnail to get a full-size plot </font>"<<endl<<endl;
	htmls[i]<<"<hr><p><p><hr>"<<endl;

	htmls[i]<<"<A HREF='"<<OutLabel<<"'>"<<endl;
	htmls[i]<<"<img src='"<<OutLabel<<"'>"<<"</A>"<<endl<<"<p>"<<endl;
      }
      else{
	htmls[i]<<"<A HREF='"<<OutLabel<<"'>"<<endl;
	htmls[i]<<"<img src='"<<OutLabel<<"' width=32%>"<<"</A>"<<endl;
      }
      if (k % 3 == 0 && k != 0) htmls[i]<<"<p>"<<endl;

      k++;
    }

    htmls[i].close();
    HistoList->close(); //delete HistoList;
  }

}

