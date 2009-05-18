#include "PhysicsTools/TagAndProbe/interface/ZLineShape.hh"
#include <RooRealVar.h>
#include <RooBifurGauss.h>
#include <RooVoigtian.h>
#include <RooAddPdf.h>

#include <iostream>

ZLineShape::ZLineShape():rooZMean_(NULL),
			 rooZWidth_(NULL),
			 rooZSigma_(NULL),
			 rooZWidthL_(NULL),
			 rooZWidthR_(NULL),
			 rooZBifurGaussFrac_(NULL),
			 rooZVoigtPdf_(NULL),
			 rooZBifurGaussPdf_(NULL)
			 
{}

ZLineShape::~ZLineShape(){
  if (rooZMean_)           delete rooZMean_;
  if (rooZWidth_)          delete rooZWidth_;
  if (rooZSigma_)          delete rooZSigma_;
  if (rooZWidthL_)         delete rooZWidthL_;
  if (rooZWidthR_)         delete rooZWidthR_;
  if (rooZBifurGaussFrac_) delete rooZBifurGaussFrac_;
  if (rooZVoigtPdf_)       delete rooZVoigtPdf_;
  if (rooZBifurGaussPdf_)  delete rooZBifurGaussPdf_;
}


void ZLineShape::Configure(const edm::ParameterSet& iConfig){

   // 1. ZLineShape
   edm::ParameterSet dLineShape;
   ZLineShape_ = iConfig.getUntrackedParameter< edm::ParameterSet >("ZLineShape",dLineShape);

   std::vector<double> dSigM;
   dSigM.push_back(91.1876);
   dSigM.push_back(85.0);
   dSigM.push_back(95.0);
   zMean_     = ZLineShape_.getUntrackedParameter< std::vector<double> >("ZMean",dSigM);
   std::vector<double> dSigW;
   dSigW.push_back(2.3);
   dSigW.push_back(1.0);
   dSigW.push_back(4.0);
   zWidth_     = ZLineShape_.getUntrackedParameter< std::vector<double> >("ZWidth",dSigW);
   std::vector<double> dSigS;
   dSigS.push_back(1.5);
   dSigS.push_back(0.0);
   dSigS.push_back(4.0);
   zSigma_     = ZLineShape_.getUntrackedParameter< std::vector<double> >("ZSigma",dSigS);
   std::vector<double> dSigWL;
   dSigWL.push_back(3.0);
   dSigWL.push_back(1.0);
   dSigWL.push_back(10.0);
   zWidthL_    = ZLineShape_.getUntrackedParameter< std::vector<double> >("ZWidthL",dSigWL);
   std::vector<double> dSigWR;
   dSigWR.push_back(0.52);
   dSigWR.push_back(0.0);
   dSigWR.push_back(2.0);
   zWidthR_    = ZLineShape_.getUntrackedParameter< std::vector<double> >("ZWidthR",dSigWR);
   std::vector<double> dBGF;
   dBGF.push_back(0.87);
   dBGF.push_back(0.0);
   dBGF.push_back(1.0);
   zBifurGaussFrac_  = ZLineShape_.getUntrackedParameter< std::vector<double> >("ZBifurGaussFrac",dBGF);
}

void  ZLineShape::CreatePdf(RooAddPdf *&ZPDF, RooRealVar *rooMass){

      // Signal PDF variables
      rooZMean_   = new RooRealVar("zMean","zMean",zMean_[0]);
      rooZWidth_  = new RooRealVar("zWidth","zWidth",zWidth_[0]);
      rooZSigma_  = new RooRealVar("zSigma","zSigma",zSigma_[0]);
      rooZWidthL_ = new RooRealVar("zWidthL","zWidthL",zWidthL_[0]);
      rooZWidthR_ = new RooRealVar("zWidthR","zWidthR",zWidthR_[0]);

      // If the user has set a range, make the variable float
      if( zMean_.size() == 3 )
      {
	 rooZMean_->setRange(zMean_[1],zMean_[2]);
	 rooZMean_->setConstant(false);
      }
      if( zWidth_.size() == 3 )
      {
	 rooZWidth_->setRange(zWidth_[1],zWidth_[2]);
	 rooZWidth_->setConstant(false);
      }
      if( zSigma_.size() == 3 )
      {
	 rooZSigma_->setRange(zSigma_[1],zSigma_[2]);
	 rooZSigma_->setConstant(false);
      }
      if( zWidthL_.size() == 3 )
      {
	 rooZWidthL_->setRange(zWidthL_[1],zWidthL_[2]);
	 rooZWidthL_->setConstant(false);
      }
      if( zWidthR_.size() == 3 )
      {
	 rooZWidthR_->setRange(zWidthR_[1],zWidthR_[2]);
	 rooZWidthR_->setConstant(false);
      }
      // Voigtian
      rooZVoigtPdf_ = new RooVoigtian("zVoigtPdf", "zVoigtPdf", 
      *rooMass, *rooZMean_, *rooZWidth_, *rooZSigma_);

      // Bifurcated Gaussian
      rooZBifurGaussPdf_ = new RooBifurGauss("zBifurGaussPdf", "zBifurGaussPdf", 
      *rooMass, *rooZMean_, *rooZWidthL_, *rooZWidthR_);
      
      // Bifurcated Gaussian fraction
      rooZBifurGaussFrac_ = new RooRealVar("zBifurGaussFrac","zBifurGaussFrac",zBifurGaussFrac_[0]);
      if( zBifurGaussFrac_.size() == 3 )
      {
	 rooZBifurGaussFrac_->setRange(zBifurGaussFrac_[1],zBifurGaussFrac_[2]);
	 rooZBifurGaussFrac_->setConstant(false);
      } 

      ZPDF = new RooAddPdf("signalShapePdf", "signalShapePdf",
      *rooZVoigtPdf_,*rooZBifurGaussPdf_,*rooZBifurGaussFrac_);      
      std::cout << ZPDF << std::endl;
      std::cout << ZPDF->getComponents()->getSize() << std::endl;
}
