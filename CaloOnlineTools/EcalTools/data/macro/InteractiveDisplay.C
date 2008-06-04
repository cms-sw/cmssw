{
#include "ToBeLoaded.C"
  string filename;
  cout<<"filename: "<<endl;
  cin >> filename;

  //string the_event; 
  //int Xtal_num; 

  int width;
  //int event = 1; 
  //cout<<"first event: "; cin>>event;
  //cout<<"Xtal: "; cin>>Xtal_num;
  cout<<"side: ";cin>>width;
  //cout<<"height: ";cin>>height;
  int height = width;
  
  int mX=width;
  int nY=width;
  
  if(!(mX>0 && mX%2 == 1 )||!(nY>0 && nY%2 == 1 )){cout<<" width and height has to be greater than 0 and odd."<<endl;return 3;}
  int* windCry = new int[mX*nY];
  H4Geom SMGeom; SMGeom.init();

  //if(Xtal_num<1 || Xtal_num > 1700){cout<<" xtal range 1->1700"<<endl;}


  int pippo = 0;

  //the_event = IntToString(event);

  TCanvas can("Digi","digi",200,50,900,900);
  can.Divide(width,height);

  TFile f(filename.c_str());
  TNtuple* eventsAndCrysNtuple = (TNtuple*) f.Get("eventsSeedCrys");
  int numEntries = eventsAndCrysNtuple->GetEntries();
  float lv1a;
  float ic;
  float fed;
  eventsAndCrysNtuple->SetBranchAddress("LV1A",&lv1a);
  eventsAndCrysNtuple->SetBranchAddress("ic",&ic);
  eventsAndCrysNtuple->SetBranchAddress("fed",&fed);

  int counter = 0;
  eventsAndCrysNtuple->GetEntry(counter);
 
  SMGeom.getWindow(windCry, ic, mX, nY);
  
  for(int kk=0 ; kk<width*height ; kk++){
    string xtal = IntToString(windCry[kk]);
    string name = "Graph_ev"+IntToString(lv1a)+"_ic";
    name+=xtal;
    name+="_FED";
    name+=IntToString(fed);
    cout<<name<<endl;
    TGraph* gra = (TGraph*) f.Get(name.c_str());
    int canvas_num = width*height - (kk%height)*width - width + 1 + kk/height;
    can.cd(canvas_num);
    if( gra != NULL ){
      gra->GetXaxis()->SetTitle("sample");
      gra->GetYaxis()->SetTitle("adc");
      gra ->Draw("A*");
      can.Update();
    }
    else
      cout << "Graph: " << name << " not found." << endl;
  }
  can.cd((width*height+1)/2);
  can.Update();
  TCanvas canB("navigation","navigation",10,50,180,200);
  TButton *but = new TButton ("Next",".x $CMSSW_BASE/src/CaloOnlineTools/EcalTools/data/macro/DrawGraphs.C",0,0,1,.5);
  but->Draw();
  TButton *butPrev = new TButton ("Prev",".x $CMSSW_BASE/src/CaloOnlineTools/EcalTools/data/macro/DrawGraphsPrev.C",0,.5,1,1);
  butPrev->Draw();
}
//return 0;
//  
//  }
