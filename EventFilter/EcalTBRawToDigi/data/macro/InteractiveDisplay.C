{
#include "ToBeLoaded.C"
  string filename;
  cout<<"filename: "<<endl;
  cin >> filename;

  TFile f(filename.c_str());
  string the_event; 
  int Xtal_num; 

  int width; 
  int height;
  int event = 1; 
  cout<<"first event: "; cin>>event;
  cout<<"Xtal: "; cin>>Xtal_num;
  cout<<"width: ";cin>>width;
  cout<<"height: ";cin>>height;

  int mX= width;
  int nY= height;
  if(!(mX>0 && mX%2 == 1 )||!(nY>0 && nY%2 == 1 )){cout<<" width and height has to be greater than 0 and odd."<<endl;return 3;}
  int* windCry = new int[mX*nY];
  H4Geom SMGeom; SMGeom.init();
  
  if(Xtal_num<1 || Xtal_num > 1700){cout<<" xtal ragne 1->1700"<<endl;}
  
  SMGeom.getWindow(windCry, Xtal_num, mX, nY);

  int pippo = 0;
  
  the_event = IntToString(event);

  TCanvas can("Digi","digi",200,50,900,900);
  can.Divide(width,height);
    
  for(int kk=0 ; kk<width*height ; kk++){
    string xtal = IntToString(windCry[kk]);
    string name = "Graph_ev"+the_event+"_ic"+xtal;
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
  }
  can.cd((width*height+1)/2);
  can.Update();
  TCanvas canB("next","next",10,50,180,200);
  TButton *but = new TButton (" next ",".x $CMSSW_BASE/src/EventFilter/EcalTBRawToDigi/data/macro/DrawGraphs.C",0,0,1,1);
  but->Draw();
}
//return 0;
  
}
