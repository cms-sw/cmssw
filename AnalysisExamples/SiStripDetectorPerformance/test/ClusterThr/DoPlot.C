#include<vector>

DoPlot(char* inputFile,string outputFile){

  gROOT->Reset();
  gStyle->("BABAR");

  gSystem->Load("DrawTree_C.so");
  TFile f(inputFile);
  f.cd("ClusterThr");
  TTree * tree=(TTree*) results;

  TCanvas C("c","c");

  std::vector<double> Tc;
  std::vector<double> Ts;
  std::vector<double> Tn;

  for (float iTc=5;iTc<13;iTc+=2)
    Tc.push_back(iTc);

  for (float iTs=3;iTs<10;iTs+=1)
    Ts.push_back(iTs);

  for (float iTn=2;iTn<6;iTn+=1)
    Tn.push_back(iTn);

  char selection[1024];

  outputFile.append("[");
  C.Print(outputFile.c_str(),"Portrait");
  outputFile.replace(outputFile.find("["),1,"");

  //pag1 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  //NTsOff:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Tn.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[j],Ts[i]);
      //cout << selection << endl;
      A.add("NTsOff:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","NTsOff");
  A.Draw(24400,31400,"Tc","NTsOff");
  C.Print(outputFile.c_str());
  
  //pag2 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //NsOff:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Tn.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[j],Ts[i]);
      //cout << selection << endl;
      A.add("NsOff:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","NsOff");
  A.Draw(0.6,0.8,"Tc","NsOff");
  C.Print(outputFile.c_str());

  //pag3 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //NTbOff:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Tn.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[j],Ts[i]);
      //cout << selection << endl;
      A.add("NTbOff:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","NTbOff");
  A.Draw(0,10000,"Tc","NTbOff");
  C.Print(outputFile.c_str());

  //pag4 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //NbOff:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Tn.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[j],Ts[i]);
      cout << selection << endl;
      A.add("NbOff:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","NbOff");
  A.Draw(0.,0.3,"Tc","NbOff");
  C.Print(outputFile.c_str());

   //pag5 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  //NTsOn:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Tn.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[j],Ts[i]);
      //cout << selection << endl;
      A.add("NTsOn:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","NTsOn");
  A.Draw(1400,1700,"Tc","NTsOn");
  C.Print(outputFile.c_str());
  
  //pag6 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //NsOn:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Tn.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[j],Ts[i]);
      //cout << selection << endl;
      A.add("NsOn:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","NsOn");
  A.Draw(0.3,1.,"Tc","NsOn");
  C.Print(outputFile.c_str());

  //pag7 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //NTbOn:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Tn.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[j],Ts[i]);
      //cout << selection << endl;
      A.add("NTbOn:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","NTbOn");
  A.Draw(0,60,"Tc","NTbOn");//TIB [0,60]
  C.Print(outputFile.c_str());

  //pag8 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //NbOn:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Tn.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[j],Ts[i]);
      cout << selection << endl;
      A.add("NbOn:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","NbOn");
  A.Draw(0.,0.1,"Tc","NbOn");
  C.Print(outputFile.c_str());

  //pag9 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  //MeanWsOff:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Tn.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[j],Ts[i]);
      cout << selection << endl;
      A.add("MeanWsOff:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","MeanWsOff");
  A.Draw(0,6,"Tc","MeanWsOff");
  C.Print(outputFile.c_str());

  //pag10 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MeanWbOff:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Tn.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[j],Ts[i]);
      //cout << selection << endl;
      A.add("MeanWbOff:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","MeanWbOff");
  A.Draw(0,6,"Tc","MeanWbOff");
  C.Print(outputFile.c_str());

  //pag11 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MeanWsOn:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Tn.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[j],Ts[i]);
      cout << selection << endl;
      A.add("MeanWsOn:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","MeanWsOn");
  A.Draw(0,6,"Tc","MeanWsOn");
  C.Print(outputFile.c_str());

  //pag12 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MeanWbOn:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Tn.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[j],Ts[i]);
      //cout << selection << endl;
      A.add("MeanWbOn:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","MeanWbOn");
  A.Draw(0,6,"Tc","MeanWbOn");
  C.Print(outputFile.c_str());

  //pag13 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  //MPVs:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Tn.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[j],Ts[i]);
      //cout << selection << endl;
      A.add("MPVs:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","MPVs");
  A.Draw(22,26,"Tc","MPVs");//TIB12 [20,24], TIB34[24,27], TOB [26,32]
  C.Print(outputFile.c_str());

  //pag14 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  //FWHMs:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Tn.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[j],Ts[i]);
      //cout << selection << endl;
      A.add("FWHMs:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","FWHMs");
  A.Draw(1,2,"Tc","FWHMs");
  C.Print(outputFile.c_str());
  
  outputFile.append("]"); 
  C.Print(outputFile.c_str());
}

