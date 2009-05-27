#include "PhysicsTools/TagAndProbe/interface/PolynomialLineShape.hh"

// CMSSW
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
// RooFit
#include <RooRealVar.h>
#include <RooAddPdf.h>
#include <RooPolynomial.h>

// STL
#include <vector>

PolynomialLineShape::PolynomialLineShape():
  rooPolyBkgC0_(NULL),
  rooPolyBkgC1_(NULL),
  rooPolyBkgC2_(NULL),
  rooPolyBkgC3_(NULL),
  rooPolyBkgC4_(NULL),
  rooPolyBkgDummyFrac_(NULL),
  rooPolyBkgPdf_(NULL),
  PolyPDF_(NULL){}

PolynomialLineShape::PolynomialLineShape(const edm::ParameterSet& polyConfig, RooRealVar *massBins):
  rooPolyBkgC0_(NULL),
  rooPolyBkgC1_(NULL),
  rooPolyBkgC2_(NULL),
  rooPolyBkgC3_(NULL),
  rooPolyBkgC4_(NULL),
  rooPolyBkgDummyFrac_(NULL),
  rooPolyBkgPdf_(NULL),
  PolyPDF_(NULL){

  Configure (polyConfig, massBins);
}

PolynomialLineShape::~PolynomialLineShape(){
  CleanUp();
}

void PolynomialLineShape::CleanUp(){
  
  if (rooPolyBkgC0_) {
    delete rooPolyBkgC0_;
    rooPolyBkgC0_ = NULL;
  }
  if (rooPolyBkgC1_) {
    delete rooPolyBkgC1_;
    rooPolyBkgC1_ = NULL;
  }
  if (rooPolyBkgC2_) {
    delete rooPolyBkgC2_;
    rooPolyBkgC2_ = NULL;
  }
  if (rooPolyBkgC3_) {
    delete rooPolyBkgC3_;
    rooPolyBkgC3_ = NULL;
  }
  if (rooPolyBkgC4_) {
    delete rooPolyBkgC4_;
    rooPolyBkgC4_ = NULL;
  }
  if (rooPolyBkgPdf_) {
    delete rooPolyBkgPdf_;
    rooPolyBkgPdf_ = NULL;
  }
  if (rooPolyBkgDummyFrac_) {
    delete rooPolyBkgDummyFrac_;
    rooPolyBkgDummyFrac_ = NULL;
  }
  if (PolyPDF_) {
    delete PolyPDF_;
    PolyPDF_ = NULL;
  }
}

void PolynomialLineShape::Configure(const edm::ParameterSet& PolynomialConfig, RooRealVar *massBins) {

  if (PolyPDF_) // If it has been configured, delete and reconfigure.
    CleanUp();

  // All parameters are off by default.
   std::vector<double> dC0;
   dC0.push_back(0.0);  
   std::vector<double>polyBkgC0_ = PolynomialConfig.getUntrackedParameter< std::vector<double> >("PolyBkgC0",dC0); 
   rooPolyBkgC0_ = new RooRealVar("polyBkgC0","polyBkgC0",polyBkgC0_[0]);
   if( polyBkgC0_.size() == 3 ) {
     rooPolyBkgC0_->setRange(polyBkgC0_[1],polyBkgC0_[2]);
     rooPolyBkgC0_->setConstant(false);
   }

   std::vector<double> dC1;
   dC1.push_back(0.0);
   std::vector<double>  polyBkgC1_ = PolynomialConfig.getUntrackedParameter< std::vector<double> >("PolyBkgC1",dC1); 
   rooPolyBkgC1_ = new RooRealVar("polyBkgC1","polyBkgC1",polyBkgC1_[0]);
   if (polyBkgC1_.size() == 3) {
     rooPolyBkgC1_->setRange(polyBkgC1_[1],polyBkgC1_[2]);
     rooPolyBkgC1_->setConstant(false);
   }

   std::vector<double> dC2;
   dC2.push_back(0.0);
   std::vector<double> polyBkgC2_ = PolynomialConfig.getUntrackedParameter< std::vector<double> >("PolyBkgC2",dC2); 
   rooPolyBkgC2_ = new RooRealVar("polyBkgC2","polyBkgC2",polyBkgC2_[0]);
   if( polyBkgC2_.size() == 3 ) {
     rooPolyBkgC2_->setRange(polyBkgC2_[1],polyBkgC2_[2]);
     rooPolyBkgC2_->setConstant(false);
   }

   std::vector<double> dC3;
   dC3.push_back(0.0);
   std::vector<double> polyBkgC3_ = PolynomialConfig.getUntrackedParameter< std::vector<double> >("PolyBkgC3",dC3); 
   rooPolyBkgC3_ = new RooRealVar("polyBkgC3","polyBkgC3",polyBkgC3_[0]);
   if( polyBkgC3_.size() == 3 ) {
     rooPolyBkgC3_->setRange(polyBkgC3_[1],polyBkgC3_[2]);
     rooPolyBkgC3_->setConstant(false);
   }
      
   std::vector<double> dC4;
   dC4.push_back(0.0);
   std::vector<double> polyBkgC4_ = PolynomialConfig.getUntrackedParameter< std::vector<double> >("PolyBkgC4",dC4); 
   rooPolyBkgC4_ = new RooRealVar("polyBkgC4","polyBkgC4",polyBkgC4_[0]);
   if( polyBkgC4_.size() == 3 ) {
     rooPolyBkgC4_->setRange(polyBkgC4_[1],polyBkgC4_[2]);
     rooPolyBkgC4_->setConstant(false);
   }

   rooPolyBkgPdf_ = new RooPolynomial("polyBkgPdf","polyBkgPdf",*massBins,
				      RooArgList(*rooPolyBkgC0_,*rooPolyBkgC1_,
						 *rooPolyBkgC2_,*rooPolyBkgC3_,*rooPolyBkgC4_),0);
   
   rooPolyBkgDummyFrac_ = new RooRealVar("dummyFrac","dummyFrac",1.0);
   
   PolyPDF_ = new RooAddPdf("bkgShapePdf", "bkgShapePdf",
			    *rooPolyBkgPdf_,*rooPolyBkgPdf_,*rooPolyBkgDummyFrac_);
}

void  PolynomialLineShape::CreatePDF (RooAddPdf *&PolyPDF) {
  if (!PolyPDF_) return;
  PolyPDF = PolyPDF_;
}
