//#include "stdlib"
//#include "stdio"
#include "iostream"
#include "vector"

std::vector< TString > labels;
std::vector<float> values;

void CreateSinglePlotFromTable(TString filename,TString XTitle="", TString YTitle=""){
  //  TString filename_=filename;
  ifstream infile;
  infile.open(filename.Data());
  if (!infile.is_open())
    return;

  cout << "filename " << filename << endl;

  char line[1024];
  int count=0;
  while (infile.good()){
    infile.getline(line,1024);
    //cout << line << endl;
    char * pch = strtok (line," ");
    while (pch != NULL){
      if (count){
	values.push_back(atof(pch));
      }else{
	labels.push_back(pch);
      }
      pch = strtok (NULL, " ");
    }
    count++;
  }
  TCanvas *c1 = new TCanvas();
  const size_t m = labels.size();
  int n = values.size()/labels.size();
  TGraph* gr[m];
  TMultiGraph *mg = new TMultiGraph();
  TLegend *tleg = new TLegend(0.9,1.,1.,0.80);
  double* x = (double*) malloc(m*n*sizeof(double));
  for (size_t i=0;i<n;++i) {
    for (size_t j=1;j<m;++j) {
    if (!i){
      gr[j-1]= new TGraph(n);
      gr[j-1]->SetMarkerStyle(19+j);
      gr[j-1]->SetMarkerColor(j);
      mg->Add(gr[j-1],"p");
      tleg->AddEntry(gr[j-1],labels[j],"p");
    }
    gr[j-1]->SetPoint(i,*(&values[0]+i*m),*(&values[0]+i*m+j));
    //cout << *(&values[0]+i*m)<< " " << *(&values[0]+i*m+j) << endl;
   }
  }
  //  c1->SetLogy();

  mg->Draw("a");
  mg->GetXaxis()->SetTitle(XTitle);
  mg->GetYaxis()->SetTitle(YTitle);
  tleg->Draw();
  gPad->Update();
  c1->Print(filename.ReplaceAll(".dat",".gif"));
};
