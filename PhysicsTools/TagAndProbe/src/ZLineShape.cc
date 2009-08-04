#include "PhysicsTools/TagAndProbe/interface/ZLineShape.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <RooRealVar.h>
#include <RooBifurGauss.h>
#include <RooVoigtian.h>
#include <RooAddPdf.h>

ZLineShape::ZLineShape ():rooZMean_(0),
			  rooZWidth_(0),
			  rooZSigma_(0),
			  rooZWidthL_(0),
			  rooZWidthR_(0),
			  rooZBifurGaussFrac_(0),
			  rooZVoigtPdf_(0),
			  rooZBifurGaussPdf_(0),
			  ZPDF_(0){}

ZLineShape::ZLineShape (const edm::ParameterSet& ZLinePSet, 
			RooRealVar *massBins):
  rooZMean_(0),
  rooZWidth_(0),
  rooZSigma_(0),
  rooZWidthL_(0),
  rooZWidthR_(0),
  rooZBifurGaussFrac_(0),
  rooZVoigtPdf_(0),
  rooZBifurGaussPdf_(0),
  ZPDF_(0){
  
  Configure(ZLinePSet, massBins);
}

ZLineShape::~ZLineShape () {
  CleanUp();
}

void ZLineShape::CleanUp(){
  if (rooZMean_) {
    delete rooZMean_;
    rooZMean_ = 0;
  }
  if (rooZWidth_) {
    delete rooZWidth_;
    rooZWidth_ = 0;
  }
  if (rooZSigma_) {
    delete rooZSigma_;
    rooZSigma_ = 0;
  }
  if (rooZWidthL_) {
    delete rooZWidthL_;
    rooZWidthL_ = 0;
  }
  if (rooZWidthR_) {
    delete rooZWidthR_;
    rooZWidthR_ = 0;
  }
  if (rooZBifurGaussFrac_) {
    delete rooZBifurGaussFrac_;
    rooZBifurGaussFrac_ = 0;
  }
  if (rooZBifurGaussPdf_) {
    delete rooZBifurGaussPdf_;
    rooZBifurGaussPdf_ = 0;
  }
  if (rooZVoigtPdf_) {
    delete rooZVoigtPdf_;
    rooZVoigtPdf_ = 0;
  }
  if (ZPDF_) {
    delete ZPDF_;
    ZPDF_ = 0;
  } 
}

void ZLineShape::Configure (const edm::ParameterSet& ZLinePSet,
			    RooRealVar *rooMass) {
  
  if (ZPDF_)
    CleanUp();

   std::vector<double> dSigM;
   dSigM.push_back(91.1876);
   dSigM.push_back(85.0);
   dSigM.push_back(95.0);
   std::vector<double> zMean_     = ZLinePSet.getUntrackedParameter< std::vector<double> >("ZMean",dSigM);
   rooZMean_   = new RooRealVar("zMean","zMean",zMean_[0]);
   if( zMean_.size() == 3 ){
     rooZMean_->setRange(zMean_[1],zMean_[2]);
     rooZMean_->setConstant(false);
   }

   std::vector<double> dSigW;
   dSigW.push_back(2.3);
   dSigW.push_back(1.0);
   dSigW.push_back(4.0);
   std::vector<double> zWidth_     = ZLinePSet.getUntrackedParameter< std::vector<double> >("ZWidth",dSigW);
   rooZWidth_  = new RooRealVar("zWidth","zWidth",zWidth_[0]);
   if( zWidth_.size() == 3 ){
     rooZWidth_->setRange(zWidth_[1],zWidth_[2]);
     rooZWidth_->setConstant(false);
   }
  
   std::vector<double> dSigS;
   dSigS.push_back(1.5);
   dSigS.push_back(0.0);
   dSigS.push_back(4.0);
   std::vector<double> zSigma_     = ZLinePSet.getUntrackedParameter< std::vector<double> >("ZSigma",dSigS);
   rooZSigma_  = new RooRealVar("zSigma","zSigma",zSigma_[0]);
   if( zSigma_.size() == 3 ){
     rooZSigma_->setRange(zSigma_[1],zSigma_[2]);
     rooZSigma_->setConstant(false);
   }

   std::vector<double> dSigWL;
   dSigWL.push_back(3.0);
   dSigWL.push_back(1.0);
   dSigWL.push_back(10.0);
   std::vector<double> zWidthL_    = ZLinePSet.getUntrackedParameter< std::vector<double> >("ZWidthL",dSigWL);
   rooZWidthL_ = new RooRealVar("zWidthL","zWidthL",zWidthL_[0]);
   if (zWidthL_.size() == 3 ) {
     rooZWidthL_->setRange(zWidthL_[1],zWidthL_[2]);
     rooZWidthL_->setConstant(false);
   }

   std::vector<double> dSigWR;
   dSigWR.push_back(0.52);
   dSigWR.push_back(0.0);
   dSigWR.push_back(2.0);
   std::vector<double> zWidthR_    = ZLinePSet.getUntrackedParameter< std::vector<double> >("ZWidthR",dSigWR);
   rooZWidthR_ = new RooRealVar("zWidthR","zWidthR",zWidthR_[0]);
   if( zWidthR_.size() == 3 ) {
     rooZWidthR_->setRange(zWidthR_[1],zWidthR_[2]);
     rooZWidthR_->setConstant(false);
   }

   std::vector<double> dBGF;
   dBGF.push_back(0.87);
   dBGF.push_back(0.0);
   dBGF.push_back(1.0);
   std::vector<double> zBifurGaussFrac_  = ZLinePSet.getUntrackedParameter< std::vector<double> >("ZBifurGaussFrac",dBGF);
   rooZBifurGaussFrac_ = new RooRealVar("zBifurGaussFrac","zBifurGaussFrac",zBifurGaussFrac_[0]);
   if( zBifurGaussFrac_.size() == 3 ) {
     rooZBifurGaussFrac_->setRange(zBifurGaussFrac_[1],zBifurGaussFrac_[2]);
     rooZBifurGaussFrac_->setConstant(false);
   } 

    // Voigtian
   rooZVoigtPdf_ = new RooVoigtian("zVoigtPdf", "zVoigtPdf", 
				   *rooMass, *rooZMean_, *rooZWidth_, *rooZSigma_);
   
   // Bifurcated Gaussian
   rooZBifurGaussPdf_ = new RooBifurGauss("zBifurGaussPdf", "zBifurGaussPdf", 
					  *rooMass, *rooZMean_, *rooZWidthL_, *rooZWidthR_);
   
   // Bifurcated Gaussian fraction   
   ZPDF_ = new RooAddPdf("signalShapePdf", "signalShapePdf",
			 *rooZVoigtPdf_,*rooZBifurGaussPdf_,*rooZBifurGaussFrac_);      
}

void  ZLineShape::CreatePDF (RooAddPdf *&ZPDF){
  
  if (!ZPDF_) return;
  ZPDF = ZPDF_;
}
