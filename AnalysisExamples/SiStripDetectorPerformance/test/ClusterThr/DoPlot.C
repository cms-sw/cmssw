DoPlot(char* inputFile,string outputFile){

  gROOT->Reset();
  gStyle->("BABAR");

  gSystem->Load("DrawTree_C.so");
  TFile f(inputFile);
  TTree * tree=(TTree*) f.Get("results");

  TCanvas C("c","c");

  std::vector<double>Tc;
  std::vector<double>Ts;
  std::vector<double>Tn;

  for (float iTc=5;iTc<15;iTc+=2)
    Tc.push_back(iTc);

  for (float iTs=3;iTs<9;iTs++)
    Ts.push_back(iTs);

  for (float iTn=2;iTn<4;iTn+=.25)
    Tn.push_back(iTn);

  char selection[1024];

  outputFile.append("[");
  C.Print(outputFile.c_str(),"Portrait");
  outputFile.replace(outputFile.find("["),1,"");
  //Ns:Tn (Tc && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(5);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Ts.size() && Ts[j]<Tc[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tc==%2.1f && Ts==%2.1f",Tc[i],Ts[j]);
      //cout << selection << endl;
      A.add("Ns:Tn",selection,"*",selection);
    }
  }
  A.setTitle("Tn","Ns");
  A.Draw();
  C.Print(outputFile.c_str());

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


  //Nb:Tn (Tc && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(5);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Ts.size() && Ts[j]<Tc[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tc==%2.1f && Ts==%2.1f",Tc[i],Ts[j]);
      //cout << selection << endl;
      A.add("Nb:Tn",selection,"*",selection);
    }
  }
  A.setTitle("Tn","Nb");
  A.Draw();
  C.Print(outputFile.c_str());


  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MeanWs:Tn (Tc && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(5);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Ts.size() && Ts[j]<Tc[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tc==%2.1f && Ts==%2.1f",Tc[i],Ts[j]);
      //cout << selection << endl;
      A.add("MeanWs:Tn",selection,"*",selection);
    }
  }
  A.setTitle("Tn","MeanWs");
  A.Draw();
  C.Print(outputFile.c_str());

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MeanWb:Tn (Tc && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(5);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Ts.size() && Ts[j]<Tc[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tc==%2.1f && Ts==%2.1f",Tc[i],Ts[j]);
      //cout << selection << endl;
      A.add("MeanWb:Tn",selection,"*",selection);
    }
  }
  A.setTitle("Tn","MeanWb");
  A.Draw();
  C.Print(outputFile.c_str());

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //Nb/Ns:Tn (Tc && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(5);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Ts.size() && Ts[j]<Tc[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tc==%2.1f && Ts==%2.1f",Tc[i],Ts[j]);
      //cout << selection << endl;
      A.add("Nb/Ns:Tn",selection,"*",selection);
    }
  }
  A.setTitle("Tn","Nb/Ns");
  A.Draw();
  C.Print(outputFile.c_str());

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  //Ns:Tn 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(4.5);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Tc==%2.1f",Tc[i]);
    //cout << selection << endl;
    A.add("Ns:Tn",selection,"*",selection);
  }
  A.setTitle("Tn","Ns");
  A.Draw();
  C.Print(outputFile.c_str());

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //Nb:Tn 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(4.5);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Tc==%2.1f && Nb<1000.",Tc[i]);
    cout << selection << endl;
    A.add("Nb:Tn",selection,"*",selection);
  }
  A.setTitle("Tn","Nb");
  A.Draw();
  C.Print(outputFile.c_str());

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MPVs:Tn 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(4.5);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Tc==%2.1f",Tc[i]);
    ////cout << selection << endl;
    A.add("MPVs:Tn",selection,"*",selection);
  }
  A.setTitle("Tn","Peak (ADC)");
  A.Draw();
  C.Print(outputFile.c_str());

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //FWHMs:Tn 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(4.5);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Tc==%2.1f",Tc[i]);
    ////cout << selection << endl;
    A.add("FWHMs:Tn",selection,"*",selection);
  }
  A.setTitle("Tn","FWHM");
  A.Draw();
  C.Print(outputFile.c_str());

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MeanWs:Tn 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(4.5);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Tc==%2.1f",Tc[i]);
    //cout << selection << endl;
    A.add("MeanWs:Tn",selection,"*",selection);
  }
  A.setTitle("Tn","Mean Width");
  A.Draw();
  C.Print(outputFile.c_str());

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MeanWb:Tn 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(4.5);
  A.setYmax(2);
  A.setYmin(0);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Tc==%2.1f",Tc[i]);
    //cout << selection << endl;
    A.add("MeanWb:Tn",selection,"*",selection);
  }
  A.Draw();
  C.Print(outputFile.c_str());

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MeanWb:Tn 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(4.5);
  A.setYmax(2);
  A.setYmin(0);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Ts==%2.1f",Ts[i]);
    //cout << selection << endl;
    A.add("MeanWb:Tn",selection,"*",selection);
  }
  A.setTitle("Tn","Mean Width");
  A.Draw();
  C.Print(outputFile.c_str());

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //RMSWs:Tn 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(4.5);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Tc==%2.1f",Tc[i]);
    //cout << selection << endl;
    A.add("RmsWs:Tn",selection,"*",selection);
  }
  A.setTitle("Tn","Rms Width");
  A.Draw();
  C.Print(outputFile.c_str());

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //Ns/Nb:Tn 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(4.5);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Tc==%2.1f",Tc[i]);
    //cout << selection << endl;
    A.add("Ns/Nb:Tn",selection,"*",selection);
  }
  A.setTitle("Tn","Ns/Nb");
  A.Draw();
  C.Print(outputFile.c_str());

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MPVs:MeanWs 
  DrawTree A(tree);
  A.setLegend(.75,.3,.1,.5);
  A.setMarkerStyle(24);
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);

    sprintf(selection,"Tc==%2.1f",Tc[i]);
    //cout << selection << endl;
    A.add("MPVs:MeanWs",selection,"*",selection);
  }
  A.setTitle("Mean Width","Peak (ADC)");
  A.Draw();
  C.Print(outputFile.c_str());

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //FWHMs:MeanWs 
  DrawTree A(tree);
  A.setLegend(.75,.3,.1,.5);
  A.setMarkerStyle(24);
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);

    sprintf(selection,"Tc==%2.1f",Tc[i]);
    //cout << selection << endl;
    A.add("FWHMs:MeanWs",selection,"*",selection);
  }
  A.setTitle("Mean Width","FWHM");
  A.Draw();
  C.Print(outputFile.c_str());

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MPVs:Ns
  DrawTree A(tree);
  A.setLegend(.75,.3,.1,.5);
  A.setMarkerStyle(24);
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);

    sprintf(selection,"Tc==%2.1f",Tc[i]);
    //cout << selection << endl;
    A.add("MPVs:Ns",selection,"*",selection);
  }
  A.setTitle("Ns","Peak (ADC)");
  A.Draw();
  C.Print(outputFile.c_str());

  outputFile.append("]"); 
  C.Print(outputFile.c_str());
}

