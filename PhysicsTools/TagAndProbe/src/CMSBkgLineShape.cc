#include "PhysicsTools/TagAndProbe/interface/CMSBkgLineShape.hh"
#include "PhysicsTools/TagAndProbe/interface/RooCMSShapePdf.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include <RooRealVar.h>
#include <RooAddPdf.h>

CMSBkgLineShape::CMSBkgLineShape():
  rooCMSBkgAlpha_(NULL),
  rooCMSBkgBeta_(NULL),
  rooCMSBkgPeak_(NULL),
  rooCMSBkgGamma_(NULL),
  rooCMSBkgDummyFrac_(NULL),
  rooCMSBkgPdf_(NULL),
  CMSBkgPDF_(NULL){}

CMSBkgLineShape::CMSBkgLineShape(const edm::ParameterSet  &pSet, 
				 RooRealVar *massBins):
  rooCMSBkgAlpha_(NULL),
  rooCMSBkgBeta_(NULL),
  rooCMSBkgPeak_(NULL),
  rooCMSBkgGamma_(NULL),
  rooCMSBkgDummyFrac_(NULL),
  rooCMSBkgPdf_(NULL),
  CMSBkgPDF_(NULL){
  
  Configure(pSet, massBins);
}

CMSBkgLineShape::~CMSBkgLineShape(){
  CleanUp();
}

void CMSBkgLineShape::CleanUp() {

  if (rooCMSBkgAlpha_) {
    delete rooCMSBkgAlpha_;
    rooCMSBkgAlpha_ = NULL;
  }
  if (rooCMSBkgBeta_) {
    delete rooCMSBkgBeta_;
    rooCMSBkgBeta_ = NULL;
  }
  if (rooCMSBkgPeak_) {
    delete rooCMSBkgPeak_;
    rooCMSBkgPeak_ = NULL;
  }
  if (rooCMSBkgGamma_) {
    delete rooCMSBkgGamma_;
    rooCMSBkgGamma_ = NULL;
  }
  if (rooCMSBkgDummyFrac_) {
    delete rooCMSBkgDummyFrac_;
    rooCMSBkgDummyFrac_ = NULL;
  }
  if (rooCMSBkgPdf_) {
    delete rooCMSBkgPdf_;
    rooCMSBkgPdf_ = NULL;
  }
  if (CMSBkgPDF_) {
    delete CMSBkgPDF_;
    CMSBkgPDF_ = NULL;
  }
}

void CMSBkgLineShape::Configure (const edm::ParameterSet& CMSBkgConfig, 
				 RooRealVar *massBins) {

  if (CMSBkgPDF_) CleanUp();

   std::vector<double> dBAl;
   dBAl.push_back(0);  // Approximate turn on from pt cut;
   std::vector<double> cmsBkgAlpha_ 
     = CMSBkgConfig.getUntrackedParameter< std::vector<double> >("CMSBkgAlpha");
   rooCMSBkgAlpha_ = new RooRealVar("cmsBkgAlpha","cmsBkgAlpha",cmsBkgAlpha_[0]);
   if (cmsBkgAlpha_.size() == 3) {
     rooCMSBkgAlpha_->setRange(cmsBkgAlpha_[1],cmsBkgAlpha_[2]);
     rooCMSBkgAlpha_->setConstant(false);
   }

   std::vector<double> dBBt;
   dBBt.push_back(0.001); // "Rise time" from pt cut;
   std::vector<double> cmsBkgBeta_ 
     = CMSBkgConfig.getUntrackedParameter< std::vector<double> >("CMSBkgBeta",dBBt);
   rooCMSBkgBeta_  = new RooRealVar("cmsBkgBeta","cmsBkgBeta",cmsBkgBeta_[0]);
   if (cmsBkgBeta_.size() == 3) {
     rooCMSBkgBeta_->setRange(cmsBkgBeta_[1],cmsBkgBeta_[2]);
     rooCMSBkgBeta_->setConstant(false);
   }

   std::vector<double> dBPk;
   dBPk.push_back(91.1876);  // Offset to extend exponetial range.
   std::vector<double> cmsBkgPeak_ 
     = CMSBkgConfig.getUntrackedParameter< std::vector<double> >("CMSBkgPeak",dBPk);
   rooCMSBkgPeak_  = new RooRealVar("cmsBkgPeak","cmsBkgPeak",cmsBkgPeak_[0]);
   if (cmsBkgPeak_.size() == 3) {
     rooCMSBkgPeak_->setRange(cmsBkgPeak_[1],cmsBkgPeak_[2]);
     rooCMSBkgPeak_->setConstant(false);
   }

   std::vector<double> dBGam; 
   dBGam.push_back(0.08);
   dBGam.push_back(0.0);
   dBGam.push_back(1.0);
   std::vector<double> cmsBkgGamma_ 
     = CMSBkgConfig.getUntrackedParameter< std::vector<double> >("CMSBkgGamma",dBGam);
   rooCMSBkgGamma_ = new RooRealVar("cmsBkgGamma","cmsBkgGamma",cmsBkgGamma_[0]);
   if( cmsBkgGamma_.size() == 3 ){
     rooCMSBkgGamma_->setRange(cmsBkgGamma_[1],cmsBkgGamma_[2]);
     rooCMSBkgGamma_->setConstant(false);
   }

   rooCMSBkgPdf_ = new RooCMSShapePdf("cmsBkgPdf","cmsBkgPdf",*massBins,*rooCMSBkgAlpha_,
				      *rooCMSBkgBeta_,*rooCMSBkgGamma_,*rooCMSBkgPeak_);
   
   rooCMSBkgDummyFrac_ = new RooRealVar("dummyFrac","dummyFrac",1.0);
   
   CMSBkgPDF_ = new RooAddPdf("bkgShapePdf", "bkgShapePdf",
			      *rooCMSBkgPdf_,*rooCMSBkgPdf_,*rooCMSBkgDummyFrac_);
   
}

void CMSBkgLineShape::CreatePDF (RooAddPdf *&CMSBkgPDF) {      

  if (!CMSBkgPDF_) return;
  CMSBkgPDF = CMSBkgPDF_;
}
