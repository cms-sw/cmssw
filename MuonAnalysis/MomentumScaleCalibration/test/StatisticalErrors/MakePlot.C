#include <iostream>
#include <fstream>
#include <sstream>
#include "TH1F.h"
#include "TCanvas.h"
#include <vector>
#include <string>

void makePlot(std::vector<double> vec, double& meanP, double& rmsP, double& sigmaP, std::string parNum);
void skimValues(std::vector<double>& vec, double mean, double rms);

void MakePlot() {
  ifstream file("Values.txt");
  std::string valueString;
  std::vector<double> values;
  bool first=true;
  std::string numPar;

  while( getline(file, valueString) ) {
    if(first) {
      numPar=valueString.substr(10,valueString.length());
      first=false;
    }
    else {
      std::stringstream ss(valueString);
      double value;
      ss >> value;
      values.push_back(value);
    }
  }
  if(file.is_open()) file.close();

  double meanPlot=0, rmsPlot=0, sigmaPlot=0;

  makePlot(values, meanPlot, rmsPlot, sigmaPlot, numPar);

  // Uncomment if you don't want "adjustments"...
  //   std::cout << "sigma_final " << sigmaPlot << std::endl;
  //   return;

  if( rmsPlot==0 || sigmaPlot==0 ) {
    std::cout << "sigma_final " << sigmaPlot << std::endl;
    return;
  }

  if( sigmaPlot/rmsPlot<5. && rmsPlot/sigmaPlot<5. ) {
    std::cout << "sigma_final " << sigmaPlot << std::endl;
    return;
  }

  std::cout << " Difference between RMS and sigma too large." << std::endl;

  int prevDim=values.size();
  int iterN=1;

  skimValues(values, meanPlot, rmsPlot);
  int lostHits=prevDim-values.size();
  int lostHitsTot=lostHits;
  prevDim=values.size();

  while( (sigmaPlot/rmsPlot<5. || rmsPlot/sigmaPlot<5.) && lostHits!=0 ) {
    std::cout << " After iteration " << iterN << ", " << lostHits << " values rejected (" << lostHitsTot << " lost)." << std::endl;
    makePlot(values, meanPlot, rmsPlot, sigmaPlot, numPar);
    skimValues(values, meanPlot, rmsPlot);
    lostHits=prevDim-values.size();
    lostHitsTot+=lostHits;
    prevDim=values.size();
    iterN++;
  }

  std::cout << "sigma_final " << sigmaPlot << std::endl;
  return;
}

void makePlot(std::vector<double> vec, double& meanP, double& rmsP, double& sigmaP, std::string parNum) {
  int cnt=vec.size();
  double minV=99999., maxV=-99999.;

  for(int kk=0; kk<cnt; ++kk) {
    if(vec[kk]<minV)
      minV=vec[kk];
    if(vec[kk]>maxV)
      maxV=vec[kk];
  }

  double minH = minV - 0.1*(maxV-minV);
  double maxH = maxV + 0.1*(maxV-minV);
  TH1F *histo = new TH1F("value", ("Parameter "+parNum).c_str(), 100, minH, maxH);
  histo->GetXaxis()->SetTitle(("Value of parameter "+parNum).c_str());
  histo->GetYaxis()->SetTitle("Entries");

  for(int i=0; i<cnt; ++i) {
    histo->Fill(vec[i]);
  }
  histo->Fit("gaus");
  TCanvas *cc = new TCanvas("cc","cc");
  histo->Draw();
  cc->SaveAs(("plot_param_"+parNum+".gif").c_str());
  meanP = histo->GetMean();
  rmsP = histo->GetRMS();
  sigmaP = ((TF1*)histo->GetListOfFunctions()->First())->GetParameter(2);
  if(histo) delete histo;
  if(cc) delete cc;
}

void skimValues(std::vector<double>& vec, double mean, double rms) {
  int dim=vec.size();
  for(int jj=0; jj<dim;) {
    double it=vec[jj];
    if( fabs( (it-mean)/rms ) > 5. ) {
      vec.erase(vec.begin()+jj);
      dim--;
    }
    else {
      jj++;
    }
  }
}
