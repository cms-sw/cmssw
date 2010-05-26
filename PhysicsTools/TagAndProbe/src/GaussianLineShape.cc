#include "PhysicsTools/TagAndProbe/interface/GaussianLineShape.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <RooRealVar.h>
#include <RooGaussian.h>
#include <RooAddPdf.h>

GaussianLineShape::GaussianLineShape():
  rooGaussMean_(0),
  rooGaussSigma_(0),
  rooGaussDummyFrac_(0),
  rooGaussPdf_(0),
  GaussPDF_(0){}

GaussianLineShape::GaussianLineShape(const edm::ParameterSet& GaussianPSet, 
				     RooRealVar *massBins):
  rooGaussMean_(0),
  rooGaussSigma_(0),
  rooGaussDummyFrac_(0),
  rooGaussPdf_(0),
  GaussPDF_(0){
  
  Configure(GaussianPSet, massBins);
}

GaussianLineShape::~GaussianLineShape(){
  CleanUp();
}

void GaussianLineShape::CleanUp() {
  if (rooGaussMean_) {
    delete rooGaussMean_;
    rooGaussMean_ = 0;
  }
  if (rooGaussSigma_) {
    delete rooGaussSigma_;
    rooGaussSigma_ = 0;
  }
  if (rooGaussPdf_) {
    delete rooGaussPdf_;
    rooGaussPdf_ = 0;
  }
  if (rooGaussDummyFrac_) {
    delete rooGaussDummyFrac_;
    rooGaussDummyFrac_ = 0;
  }
  if (GaussPDF_) {
    delete GaussPDF_;
    GaussPDF_ = 0;
  }
}

void GaussianLineShape::Configure (const edm::ParameterSet& GaussianPSet, 
				   RooRealVar *massBins) {
  
  if (GaussPDF_) 
    CleanUp();
  
  std::vector<double> gaussMean_ = 
    GaussianPSet.getUntrackedParameter< std::vector<double> >("GaussMean");
  
  std::vector<double> gaussSigma_ = 
    GaussianPSet.getUntrackedParameter< std::vector<double> >("GaussSigma");
  
  // Signal PDF variables
  rooGaussMean_  = new RooRealVar("gaussMean","gaussMean",gaussMean_[0]);
  rooGaussSigma_ = new RooRealVar("gaussSigma","gaussSigma",gaussSigma_[0]);
  
  // If the user has set a range, make the variable float
  if( gaussMean_.size() == 3 ){
    rooGaussMean_->setRange(gaussMean_[1],gaussMean_[2]);
    rooGaussMean_->setConstant(false);
  }
  if( gaussSigma_.size() == 3 ){
    rooGaussSigma_->setRange(gaussSigma_[1],gaussSigma_[2]);
    rooGaussSigma_->setConstant(false);
  }
  rooGaussDummyFrac_ = new RooRealVar("dummyFrac","dummyFrac",1.0);
  
  rooGaussPdf_ = new RooGaussian("gaussPdf","gaussPdf",*massBins,*rooGaussMean_,
				 *rooGaussSigma_);
  
  GaussPDF_ = new RooAddPdf("signalShapePdf", "signalShapePdf",
			    *rooGaussPdf_,*rooGaussPdf_,*rooGaussDummyFrac_);
}

void GaussianLineShape::CreatePDF(RooAddPdf *&GaussPDF){
  
  if (!GaussPDF_) return;
  GaussPDF = GaussPDF_;
}
