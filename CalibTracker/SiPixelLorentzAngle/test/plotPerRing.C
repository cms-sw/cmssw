{
  //gROOT->Reset();
  gROOT->ProcessLine(".L ./style-CMSTDR.C");
  gROOT->ProcessLine("setTDRStyle()");
  char name[256];
  float rings[8]={1,2,3,4,5,6,7,8};
  float errRings[8]={0,0,0,0,0,0,0,0};

  //////////////
  sprintf(name,"layer1.txt");
  Int_t k=0;
  float L1angle[8];
  float L1error[8];
  ifstream infile;
  infile.open(name);
  if(!infile)
    {
      cout<<"Can not open input file"<<endl;
      exit(1);
    }
  int i=0;
  while(!infile.eof())
    {
      infile>>L1angle[i]>>L1error[i];
      i++;
      k++;
    }
  infile.close();

  for(int i=0;i<8;++i){
    cout<<L1angle[i]<<"\t"<<L1error[i]<<endl;
  }

  //////////////////
  
  //////////////
  sprintf(name,"layer2.txt");
  Int_t k=0;
  float L2angle[8];
  float L2error[8];
  ifstream infile;
  infile.open(name);
  if(!infile)
    {
      cout<<"Can not open input file"<<endl;
      exit(1);
    }
  int i=0;
  while(!infile.eof())
    {
      infile>>L2angle[i]>>L2error[i];
      i++;
      k++;
    }
  infile.close();

  for(int i=0;i<8;++i){
    cout<<L2angle[i]<<"\t"<<L2error[i]<<endl;
  }

  //////////////////

  //////////////
  sprintf(name,"layer3.txt");
  Int_t k=0;
  float L3angle[8];
  float L3error[8];
  ifstream infile;
  infile.open(name);
  if(!infile)
    {
      cout<<"Can not open input file"<<endl;
      exit(1);
    }
  int i=0;
  while(!infile.eof())
    {
      infile>>L3angle[i]>>L3error[i];
      i++;
      k++;
    }
  infile.close();

  for(int i=0;i<8;++i){
    cout<<L3angle[i]<<"\t"<<L3error[i]<<endl;
  }

  //////////////////

 

  //create graphs
  TMultiGraph *mg = new TMultiGraph();
  TCanvas *c1 = new TCanvas("c1","Graph Draw Options",200,10,800,600);
  TGraph *gr1 = new TGraphErrors(8,rings,L1angle,errRings,L1error);
  TGraph *gr2 = new TGraphErrors(8,rings,L2angle,errRings,L2error);
  TGraph *gr3 = new TGraphErrors(8,rings,L3angle,errRings,L3error);

  gr1->SetMarkerColor(2);
  gr1->SetMarkerSize(1.3);
  gr1->SetMarkerStyle(20);
  gr1->SetLineColor(2);
  gr1->SetLineWidth(2);

  gr2->SetMarkerColor(8);
  gr2->SetMarkerSize(1.3);
  gr2->SetMarkerStyle(22);
  gr2->SetLineColor(3);
  gr2->SetLineWidth(2);

  gr3->SetMarkerColor(4);
  gr3->SetMarkerSize(1.3);
  gr3->SetMarkerStyle(23);
  gr3->SetLineColor(4);
  gr3->SetLineWidth(2);
  mg->Add(gr3);
  mg->Add(gr2);
  mg->Add(gr1);

  mg->Draw("AP");
  mg->GetYaxis()->SetRangeUser(0.36,0.44);
  mg->GetXaxis()->SetTitle("Ring");
  mg->GetYaxis()->SetTitle("tan#theta_{LA}");
 

  //draw the mean LA value as a line
  TLine mean(0.65,0.4081,8.37,0.4081);
  mean.SetLineWidth(3);
  mean.Draw();

  leg = new TLegend(0.6,0.7,0.89,0.89);  //coordinates are fractions
                                         //of pad dimensions
  leg->AddEntry(gr1,"layer 1","p");  // "l" means line
  leg->AddEntry(gr2,"layer 2","p");
  leg->AddEntry(gr3,"layer 3","p"); // use "f" for a box
  leg->AddEntry(&mean,"mean value","l");
  leg->SetFillColor(0);
  leg->SetBorderSize(0);
  leg->Draw();


}//end
