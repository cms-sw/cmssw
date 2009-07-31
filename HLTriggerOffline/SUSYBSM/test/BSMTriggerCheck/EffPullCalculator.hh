#ifndef EFFPULLCALC_HH
#define EFFPULLCALC_HH

class EffPullcalculator {
public:
  EffPullcalculator(TH1D* pathHisto1_v, TH1D* pathHisto2_v, vector<TH1D*> sortedHisto1_v, vector<TH1D*> sortedHisto2_v, string error_v);
  ~EffPullcalculator(){};
  
  TH1D*  GetResidualHisto() {return resHisto;}
  TH1D*  GetPullHisto() {return pullDist;}
  vector<TH1D*>  GetEffHistos() {return effhisto;}
  void   CalculatePulls();
  double GetEff(string label, int ind);
  void   WriteLogFile(string namefile);
  vector<int> SortVec(vector<double> eff);
  void AddGoldenPath(string name);
  void PrintTwikiTable(string filename);
  bool GoodLabel(string pathname);
  double abs(double value);

private:
  // log file
  vector<string> lines;
  vector<TH1D*> pathHisto;
  vector<TH1D*> effhisto;
  vector<TH1D*> sortedHisto1;
  vector<TH1D*> sortedHisto2;
  TH1D* pullHisto;
  TH1D* pullDist;
  TH1D* resHisto;
  vector<double> eff1;
  vector<double> eff2;
  vector<double> err_eff1;
  vector<double> err_eff2;
  vector<string> name;
  string error;

  // golden paths for twiki tables
  vector<string> goldenpaths;
};

#endif
