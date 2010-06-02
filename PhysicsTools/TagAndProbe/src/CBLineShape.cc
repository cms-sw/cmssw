#include "PhysicsTools/TagAndProbe/interface/CBLineShape.hh"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <RooRealVar.h>
#include <RooCBShape.h>
#include <RooAddPdf.h>

#include <vector>

CBLineShape::CBLineShape():rooCBMean_(0),
			   rooCBSigma_(0),
			   rooCBAlpha_(0),
			   rooCBN_(0),
			   rooCBDummyFrac_(0),
			   rooCBPdf_(0),
			   CBPDF_(0){}

CBLineShape::CBLineShape(const edm::ParameterSet& CBLinePSet,
			 RooRealVar *massBins):
  rooCBMean_(0),
  rooCBSigma_(0),
  rooCBAlpha_(0),
  rooCBN_(0),
  rooCBDummyFrac_(0),
  rooCBPdf_(0),
  CBPDF_(0){

  Configure(CBLinePSet, massBins);
}

CBLineShape::~CBLineShape(){
  CleanUp();
}

void CBLineShape::CleanUp() {

  if (rooCBMean_) {
    delete rooCBMean_;
    rooCBMean_ = 0;
  }
  if (rooCBSigma_) {
    delete rooCBSigma_;
    rooCBSigma_ = 0;
  }
  if (rooCBAlpha_) {
    delete rooCBAlpha_;
    rooCBAlpha_ = 0;
  }
  if (rooCBN_) {
    delete rooCBN_;
    rooCBN_ = 0;
  }
  if (rooCBDummyFrac_) {
    delete rooCBDummyFrac_;
    rooCBDummyFrac_ = 0;
  }
  if (rooCBPdf_) {
    delete rooCBPdf_;
    rooCBPdf_ = 0;
  }
  if (CBPDF_) {
    delete CBPDF_;
    CBPDF_ = 0;
  }
}

void CBLineShape::Configure(const edm::ParameterSet& CBLineShape_, RooRealVar *massBins){

  if (CBPDF_) CleanUp();

  std::vector<double> dCBM;
  dCBM.push_back(0.0);
  std::vector<double> cbMean_          = CBLineShape_.getUntrackedParameter< std::vector<double> >("CBMean",dCBM);
  rooCBMean_  = new RooRealVar("cbMean","cbMean",cbMean_[0]);
  if( cbMean_.size() == 3 ) {
    rooCBMean_->setRange(cbMean_[1],cbMean_[2]);
    rooCBMean_->setConstant(false);
  }
  
  std::vector<double> dCBS;
  dCBS.push_back(0.0);
  std::vector<double>cbSigma_         = CBLineShape_.getUntrackedParameter< std::vector<double> >("CBSigma",dCBS);
  rooCBSigma_ = new RooRealVar("cbSigma","cbSigma",cbSigma_[0]);
  if( cbSigma_.size() == 3 ) {
    rooCBSigma_->setRange(cbSigma_[1],cbSigma_[2]);
    rooCBSigma_->setConstant(false);
  }
  
  std::vector<double> dCBAlpha;
  dCBAlpha.push_back(0.0);
  std::vector<double>cbAlpha_         = CBLineShape_.getUntrackedParameter< std::vector<double> >("CBAlpha",dCBAlpha);
  rooCBAlpha_ = new RooRealVar("cbAlpha","cbAlpha",cbAlpha_[0]);
  if( cbAlpha_.size() == 3 ) {
    rooCBAlpha_->setRange(cbAlpha_[1],cbAlpha_[2]);
    rooCBAlpha_->setConstant(false);
  }
  
  std::vector<double> dCBN;
  dCBN.push_back(0.0);
  std::vector<double>cbN_             = CBLineShape_.getUntrackedParameter< std::vector<double> >("CBN",dCBN);
  rooCBN_     = new RooRealVar("cbN","cbN",cbN_[0]);
  if( cbN_.size() == 3 ) {
    rooCBN_->setRange(cbN_[1],cbN_[2]);
    rooCBN_->setConstant(false);
  }


  
  rooCBDummyFrac_ = new RooRealVar("dummyFrac","dummyFrac",1.0);
  
  
  rooCBPdf_ = new RooCBShape("cbPdf","cbPdf",*massBins,*rooCBMean_,
			     *rooCBSigma_,*rooCBAlpha_,*rooCBN_);
  
  CBPDF_ = new RooAddPdf("signalShapePdf", "signalShapePdf",
			 *rooCBPdf_,*rooCBPdf_,*rooCBDummyFrac_);

}

void CBLineShape::CreatePDF(RooAddPdf *&CBPDF){
  
  if (!CBPDF_) return;
  CBPDF = CBPDF_;
}
