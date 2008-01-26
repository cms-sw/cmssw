DoPlot(char* inputFile,string outputFile){

  gROOT->Reset();
  gStyle->("BABAR");

  gSystem->Load("DrawTree_C.so");
  TFile f(inputFile);
  f.cd("AsciiOutput");
  TTree * tree=(TTree*) results;

  TCanvas C("c","c");

  std::vector<double> Tc;
  std::vector<double> Ts;
  std::vector<double> Tn;

 for (float iTc=5;iTc<11;iTc+=0.5)
    Tc.push_back(iTc);

  for (float iTs=2.5;iTs<5;iTs+=0.25)
    Ts.push_back(iTs);

  for (float iTn=1.5;iTn<2.5;iTn+=.25)
    Tn.push_back(iTn);

  char selection[1024];

  outputFile.append("[");
  C.Print(outputFile.c_str(),"Portrait");
  outputFile.replace(outputFile.find("["),1,"");

  //pag1 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  //NTs:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Tn.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Ts.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[i],Ts[j]);
       //cout << selection << endl;
      A.add("NTs:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","NTs");
  A.Draw();
  C.Print(outputFile.c_str());
  
  //pag2 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //Ns:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Tn.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Ts.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[i],Ts[j]);
      //cout << selection << endl;
      A.add("Ns:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","Ns");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag3 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //NTb:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Tn.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Ts.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[i],Ts[j]);
      //cout << selection << endl;
      A.add("NTb:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","NTb");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag4 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //Nb:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Tn.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Ts.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[i],Ts[j]);
      //cout << selection << endl;
      A.add("Nb:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","Nb");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag5 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MeanWs:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Tn.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Ts.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[i],Ts[j]);
      //cout << selection << endl;
      A.add("MeanWs:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","MeanWs");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag6 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MeanWb:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Tn.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Ts.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[i],Ts[j]);
      //cout << selection << endl;
      A.add("MeanWb:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","MeanWb");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag7 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //NTb/NTs:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(12);
  int val=19;
  for (size_t i=0;i<Tn.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Ts.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[i],Ts[j]);
      //cout << selection << endl;
      A.add("NTb/NTs:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","NTb/NTs");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag 8 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //Nb/Ns:Tc (Tn && Ts)
  DrawTree A(tree);
  A.setLegend(.7,.2,.1,.7);
  A.setXmax(5);
  int val=19;
  for (size_t i=0;i<Tn.size() ;i++){
    A.setMarkerColor(1);
    val++;
    for (size_t j=0;j<Ts.size() && Tn[j]<Ts[i];j++){    
      A.setMarkerStyle(val);
      sprintf(selection,"Tn==%2.2f && Ts==%2.2f",Tn[i],Ts[j]);
      //cout << selection << endl;
      A.add("Nb/Ns:Tc",selection,"*",selection);
    }
  }
  A.setTitle("Tc","Nb/Ns");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag 9 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  //NTs:Tn 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(3.0);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Tc==%2.2f",Tc[i]);
    //cout << selection << endl;
    A.add("NTs:Tn",selection,"*",selection);
  }
  A.setTitle("Tn","NTs");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag 10 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  //Ns:Tn 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(3.0);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Tc==%2.2f",Tc[i]);
    //cout << selection << endl;
    A.add("Ns:Tn",selection,"*",selection);
  }
  A.setTitle("Tn","Ns");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag 11 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //NTb:Tn 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(3.0);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Tc==%2.2f && NTb<1000.",Tc[i]);
    cout << selection << endl;
    A.add("NTb:Tn",selection,"*",selection);
  }
  A.setTitle("Tn","NTb");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag 12 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MPVs:Ts 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(4.5);
  int val=19;
  for (size_t i=0;i<Tn.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Tn==%2.2f",Tn[i]);
    ////cout << selection << endl;
    A.add("MPVs:Ts",selection,"*",selection);
  }
  A.setTitle("Ts","S/N MPV");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag 13 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //FWHMs:Tc 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(4.5);
  int val=19;
  for (size_t i=0;i<Tn.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Tn==%2.2f",Tn[i]);
    ////cout << selection << endl;
    A.add("FWHMs:Tc",selection,"*",selection);
  }
  A.setTitle("Tc","FWHM");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag 14 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MeanWs:Tn 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(3.0);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Ts==%2.2f",Ts[i]);
    //cout << selection << endl;
    A.add("MeanWs:Tn",selection,"*",selection);
  }
  A.setTitle("Tn","Mean S Width");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag 15 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
 
  //MeanWb:Tn 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(3.0);
  A.setYmax(2);
  A.setYmin(0);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Ts==%2.2f",Ts[i]);
    //cout << selection << endl;
    A.add("MeanWb:Tn",selection,"*",selection);
  }
  A.setTitle("Tn","Mean BG Width");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag 16 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MPVs:Tn 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(3.0);
  A.setYmax(2);
  A.setYmin(0);
  int val=19;
  for (size_t i=0;i<Ts.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Ts==%2.2f",Ts[i]);
    //cout << selection << endl;
    A.add("MPVs:Tn",selection,"*",selection);
  }
  A.setTitle("Tn","S/N MPV");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag 17 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //RMSWs:Tn 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(3.0);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Tc==%2.2f",Tc[i]);
    //cout << selection << endl;
    A.add("RmsWs:Tn",selection,"*",selection);
  }
  A.setTitle("Tn","Rms Width");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag 18 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //NTb/NTs:Ts 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(4.5);
  int val=19;
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Tc==%2.2f",Tc[i]);
    //cout << selection << endl;
    A.add("NTb/NTs:Ts",selection,"*",selection);
  }
  A.setTitle("Ts","NTb/NTs");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag 19 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //Nb/Ns:Ts 
  DrawTree A(tree);
  A.setLegend(.75,.5,.1,.5);
  A.setXmax(4.5);
  int val=19;
  for (size_t i=0;i<Tn.size() ;i++){
    A.setMarkerColor(1);
    val++;
    A.setMarkerStyle(val);
    sprintf(selection,"Tc==%2.2f",Tc[i]);
    //cout << selection << endl;
    A.add("Nb/Ns:Ts",selection,"*",selection);
  }
  A.setTitle("Ts","Nb/Ns");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag 20 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MPVs:MeanWs 
  DrawTree A(tree);
  A.setLegend(.75,.3,.1,.5);
  A.setMarkerStyle(24);
  for (size_t i=0;i<Tn.size() ;i++){
    A.setMarkerColor(1);

    sprintf(selection,"Tn==%2.2f",Tn[i]);
    //cout << selection << endl;
    A.add("MPVs:MeanWs",selection,"*",selection);
  }
  A.setTitle("Mean S Width","Peak");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag 21 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //FWHMs:MeanWs 
  DrawTree A(tree);
  A.setLegend(.75,.3,.1,.5);
  A.setMarkerStyle(24);
  for (size_t i=0;i<Tn.size() ;i++){
    A.setMarkerColor(1);

    sprintf(selection,"Tn==%2.2f",Tn[i]);
    //cout << selection << endl;
    A.add("FWHMs:MeanWs",selection,"*",selection);
  }
  A.setTitle("Mean S Width","FWHM");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag 22 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MPVs:NTs
  DrawTree A(tree);
  A.setLegend(.75,.3,.1,.5);
  A.setMarkerStyle(24);
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);

    sprintf(selection,"Tc==%2.2f",Tc[i]);
    //cout << selection << endl;
    A.add("MPVs:NTs",selection,"*",selection);
  }
  A.setTitle("NTs","S/N MPV");
  A.Draw();
  C.Print(outputFile.c_str());

  //pag 23 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //MPVs:Ns
  DrawTree A(tree);
  A.setLegend(.75,.3,.1,.5);
  A.setMarkerStyle(24);
  for (size_t i=0;i<Tc.size() ;i++){
    A.setMarkerColor(1);

    sprintf(selection,"Tc==%2.2f",Tc[i]);
    //cout << selection << endl;
    A.add("MPVs:Ns",selection,"*",selection);
  }
  A.setTitle("Ns","S/N MPV");
  A.Draw();
  C.Print(outputFile.c_str());

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

   outputFile.append("]"); 
   C.Print(outputFile.c_str());
}

