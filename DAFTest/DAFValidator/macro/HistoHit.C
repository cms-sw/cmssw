
void drawingPlotFromOneFile(const std::string& datafile)
{
  float x[20] = {0.0, 0.5, 1, 2.5, 5, 10, 25, 50, 100, 10000, 1000000, 10000000, 50000000, 75000000, 100000000, 250000000, 500000000, 750000000, 1000000000, 5000000000};
  TH1F *g = new TH1F("g","histo",19,x);//100,0.5,100.5);//,1,0.5,10000000000);
  std::string perc;
  std::string line;
  int i;
  int npar = 5;
  std::ifstream inFile;
  inFile.open (datafile.c_str(), std::ios::in);

  TF1 *f1 = new TF1("f1","sin(x)/x",-10,10);

  //con parametri esterni
  //TF1 *f1 = new TF1("f1","[0]*x*sin([1]*x)",-3,3);
  //f1->SetParameter(0,10);
  //f1->SetParameter(1,2);

  f1->SetTitle("Disegno di una funzione");
  f1->GetXaxis ()->SetTitle ("x");
  f1->GetYaxis ()->SetTitle ("sin(x)/x");
  f1->GetXaxis ()->CenterTitle ();
  f1->GetYaxis ()->CenterTitle ();


  while(!inFile.eof())
  {
    getline(inFile, line);
    istringstream iss(line);
    string sub;
    iss >> sub;
    if (sub == "Begin" || sub == "something"){
      std::cin.ignore(256,' ');
    } else {
      inFile >> perc;
      if(atof(perc.c_str())) {
        double value = (double)atof(perc.c_str());
        //std::cout<<"Alla riga " <<i<<" ho letto " << value <<std::endl;
        g->Fill(value*npar);
      }
    }
    i++;

  }
  
  inFile.close();

  TCanvas *c = new TCanvas("c_graph","c_graph");
  c->cd();
  c->SetGridy();
  c->SetGridx();
  c->SetTickx();
  c->SetTicky();
  c->SetLogy();
  c->SetLogx();

  g->SetTitle("TGraph Title");
  g->GetXaxis()->SetTitle("#chi^{2}");
  g->GetYaxis()->SetTitle("N");
  g->SetMarkerStyle(11);
  g->SetMarkerColor(6);
//  g->SetMinimum(0);
//  g->SetMaximum(40);
  g->Draw("HISTO");
  f1->Draw("same");

  c->Print( "histoProva.pdf","pdf" );

}


void drawingPlotFromTwoFile(const std::string& datafile1, const std::string& datafile2)
{
  TH1F *g = new TH1F("g","chi2",30,0.5,30.5);//,1,0.5,10000000000);
  TH1F *g2= new TH1F("g2","chi2_prob",50,0.0,1.0);//,1,0.5,10000000000);
  TH1F *h = new TH1F("h","histo2",30,0.5,30.5);
  TH1F *h2= new TH1F("h2","chi2_prob2",50,0.0,1.0);//,1,0.5,10000000000);
  std::string perc;
  std::string line;
  int i;
  int npar = 5;
  double chi2Cut = 100;
  std::ifstream inFile1, inFile2;
  inFile1.open (datafile1.c_str(), std::ios::in);
  inFile2.open (datafile2.c_str(), std::ios::in);

  TF1 *fgamma = new TF1("fgamma", "TMath::GammaDist(x, [0], [1], [2])", 0, 30);
  fgamma->SetParameters(npar/2, 0, 2);


  while(!inFile1.eof())
  {
    getline(inFile1, line);
    istringstream iss(line);
    string sub;
    iss >> sub;
    if (sub == "Begin" || sub == "something"){
      std::cin.ignore(256,' ');
    } else {
      inFile1 >> perc;
      if(atof(perc.c_str())) {
        double value = (double)atof(perc.c_str());
        //std::cout<<"Alla riga " <<i<<" ho letto " << value <<std::endl;
        if(value < chi2Cut) g->Fill(value*npar);

          g2->Fill(TMath::Prob(value*npar, npar));

      }
    }
    i++;

  }

  inFile1.close();
  i = 0;

  while(!inFile2.eof())
  {
    getline(inFile2, line);
    istringstream iss(line);
    string sub;
    iss >> sub;
    if (sub == "Begin" || sub == "something"){
      std::cin.ignore(256,' ');
    } else {
      inFile2 >> perc;
      if(atof(perc.c_str())) {
        double value = (double)atof(perc.c_str());
        //std::cout<<"Alla riga " <<i<<" ho letto " << value <<std::endl;
        if(value < chi2Cut)  h->Fill(value*npar);

	  h2->Fill(TMath::Prob(value*npar, npar));

      }
    }
    i++;

  }
  inFile2.close();

  TCanvas *c = new TCanvas("c_graph","c_graph");
  c->cd();
  c->SetGridy();
  c->SetGridx();
  c->SetTickx();
  c->SetTicky();
//  c->SetLogy();

  TLegend* legend = new TLegend(0.56, 0.77, 0.76, 0.92);
  legend -> SetFillColor(kWhite);
  legend -> SetFillStyle(1001);
  legend -> SetTextFont(42);
  legend -> SetTextSize(0.03);

  legend -> AddEntry(g,"cutsRecoTracks","L");
  legend -> AddEntry(h,"ctfWithMaterialTracksDAF","L");
  legend -> AddEntry(fgamma,"#chi^{2} with 5 dof","l");

  g->SetTitle("TGraph Title");
  g->GetXaxis()->SetTitle("#chi^{2}");
  g->GetYaxis()->SetTitle("N/N_{tot}");
  g->SetMarkerStyle(11);
  g->SetMarkerColor(6);
  g->SetLineColor(2);
  g -> GetXaxis() -> SetTitleSize(0.03);
  g -> GetYaxis() -> SetTitleSize(0.03);
  g -> GetXaxis() -> SetLabelSize(0.03);
  g -> GetYaxis() -> SetLabelSize(0.03);
  h->SetLineColor(4);
  h->SetLineWidth(2);
// g->Fit(fgamma);
  fgamma->SetLineColor(13);
  fgamma->SetLineWidth(2);
  fgamma->Draw();
  g->DrawNormalized("sameHISTO");
  h->DrawNormalized("sameHISTO");
  legend->Draw("same");
  fgamma->SetLineColor(13);
  fgamma->SetLineWidth(2);
  fgamma->Draw("same");

  c->Print( "histoProva2.pdf","pdf" );

  legend->Clear();
  legend -> AddEntry(g,"cutsRecoTracks","L");
  legend -> AddEntry(h,"ctfWithMaterialTracksDAF","L");

  TCanvas *c2 = new TCanvas("c_graph2","c_graph2");
  c2->cd();
  c2->SetGridy();
  c2->SetGridx();
  c2->SetTickx();
  c2->SetTicky();
  //c2->SetLogy();
  g2->GetXaxis()->SetTitle("#chi^{2} prob");
  g2->GetYaxis()->SetTitle("N");
  g2-> GetXaxis() -> SetTitleSize(0.03);
  g2-> GetYaxis() -> SetTitleSize(0.03);
  g2-> GetXaxis() -> SetLabelSize(0.03);
  g2-> GetYaxis() -> SetLabelSize(0.03);
  g2->SetLineColor(2);
  h2->SetLineColor(4);
  h2->SetLineWidth(2);
  g2->Draw();
  h2->Draw("same");  
  legend->Draw("same");
  c->Print( "histoProva3.pdf","pdf" );
}

