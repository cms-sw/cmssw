#ifndef COMP1D_HH
#define COMP1D_HH

class CompHisto1D {
public:
  CompHisto1D(TH1D* histo1_v, TH1D* histo2_v);
  ~CompHisto1D(){};

  double Compare();
  void SaveAsEps();
  void SetLabel1(string l1) {label1 = l1;}
  void SetLabel2(string l2) {label2 = l2;}
  double myChisq();  

private:
  TH1D* histo1;
  TH1D* histo2;
  string title;
  string label1;
  string label2;
  TCanvas* compCanvas;

};

#endif
