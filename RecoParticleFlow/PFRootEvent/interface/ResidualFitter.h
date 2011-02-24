#ifndef RecoParticleFlow_PFRootEvent_ResidualFitter_h
#define RecoParticleFlow_PFRootEvent_ResidualFitter_h

#include <RQ_OBJECT.h>
#include <TH3.h>
#include <TH2.h>
#include <TCanvas.h>
#include <TF1.h>


class ResidualFitter : public TH3D, public TQObject {

  //  RQ_OBJECT("ResidualFitter")

 public:
  ResidualFitter(const char* name, 
                 const char* title, 
                 int nbinsx, double xlow, double xup, 
                 int nbinsy, double ylow, double yup, 
                 int nbinsz, double zlow, double zup);

  ~ResidualFitter();

  static void SetCanvas(int x, int y) { xCanvas_=x; yCanvas_=y;}
  void CreateCanvas();

  void AddTrue( double x, double y, double z) 
    {true_.push_back( Point(x,y,z) ); }
  void AddMeas( double x, double y, double z )
    {meas_.push_back( Point(x,y,z) ); }
  void Fill();

  void SetFitFunction(TF1* func) { fitFunction_ = func; }  
  void SetFitOptions(const char* opt) {fitOptions_ = opt; }

  void FitSlicesZ(TF1 *f1=0);
  void ExecuteEvent(Int_t event, Int_t px, Int_t py, TObject *sel);

  void Fit(int x, int y, const char* opt=""); 
  void ShowFit(TH2* h2d, double x, double y);
  
  void SetAutoRange(int n) {
    autoRangeN_= n;
    fitOptions_ += "R";
  }

  void SetMinN(int n) { minN_ = n;}

  void cd() {canvas_->cd(); }

  TF1* GetFunction() {return fitFunction_;}

  TH1D* CurBin() {return curBin_;}
  
  TCanvas* GetCanvas() {return canvas_;}

  class Point {
  public:
    Point() : x_(-10), y_(-10), z_(-10) {}
    Point(double x, double y, double z) :
      x_(x), y_(y), z_(z) {} 
  private:
    double x_;
    double y_;
    double z_;    
    
    friend class ResidualFitter;
  };

 private:
  std::vector< ResidualFitter::Point > true_;
  std::vector< ResidualFitter::Point > meas_;
  
  TF1*          fitFunction_;
  std::string   fitOptions_;

  TCanvas*  canvas_;
  
  static int    xCanvas_;
  static int    yCanvas_;
  

  TCanvas*  canvasFit_;

  TH1D*     curBin_;
  TH2D*     mean_;
  TH2D*     sigma_; 
  TH2D*     chi2_;
  /*   TH2D*     nseen_; */

  int       oldBinx_;
  int       oldBiny_;

  int       autoRangeN_;

  /// if the number of entries in a bin is lower than this, fit
  /// is not performed. set to 5 by default.
  int       minN_;

  ClassDef(ResidualFitter,1)
};

#endif
