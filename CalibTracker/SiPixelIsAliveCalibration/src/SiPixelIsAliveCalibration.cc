// -*- C++ -*-
//
// Package:    SiPixelIsAliveCalibrationAnalyzer
// Class:      SiPixelIsAliveCalibrationAnalyzer
// 
/**\class SiPixelIsAliveCalibration SiPixelIsAliveCalibration.cc CalibTracker/SiPixelIsAliveCalibration/src/SiPixelIsAliveCalibration.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Freya Blekman
//         Created:  Mon Dec  3 14:07:42 CET 2007
// $Id: SiPixelIsAliveCalibration.cc,v 1.21 2008/09/18 11:25:17 krose Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibTracker/SiPixelTools/interface/SiPixelOfflineCalibAnalysisBase.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelCalibDigi.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//
// class decleration
//

class SiPixelIsAliveCalibration : public SiPixelOfflineCalibAnalysisBase {
   public:
      explicit SiPixelIsAliveCalibration(const edm::ParameterSet&);
      ~SiPixelIsAliveCalibration();

  void doSetup(const edm::ParameterSet &);
      virtual bool doFits(uint32_t detid, std::vector<SiPixelCalibDigi>::const_iterator ipix);
  

   private:
  
      virtual void calibrationSetup(const edm::EventSetup&) ;
      virtual void calibrationEnd() ;
  virtual void newDetID(uint32_t detid);
  virtual bool checkCorrectCalibrationType();
      // ----------member data ---------------------------

  std::map<uint32_t,MonitorElement *> bookkeeper_;
  std::map<uint32_t,MonitorElement *> summaries_;
  double mineff_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SiPixelIsAliveCalibration::SiPixelIsAliveCalibration(const edm::ParameterSet& iConfig):
  SiPixelOfflineCalibAnalysisBase(iConfig),
  mineff_(iConfig.getUntrackedParameter<double>("minEfficiencyForAliveDef",0.8))
{
   //now do what ever initialization is needed

}


SiPixelIsAliveCalibration::~SiPixelIsAliveCalibration()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//
void 
SiPixelIsAliveCalibration::newDetID(uint32_t detid){
  setDQMDirectory(detid);
  std::string tempname=translateDetIdToString(detid);
  bookkeeper_[detid]= bookDQMHistoPlaquetteSummary2D(detid,"pixelAlive","pixel alive for "+tempname); 
  int xpix = bookkeeper_[detid]->getNbinsX();
  int ypix = bookkeeper_[detid]->getNbinsY();
  int tpix = xpix*ypix;
  summaries_[detid]= bookDQMHistogram1D(detid,"pixelAliveSummary",bookkeeper_[detid]->getTitle(),calib_->getNTriggers()+1,-.05,.95+(1./(float)calib_->getNTriggers()));
  summaries_[detid]->setBinContent(1, tpix);
}
bool
SiPixelIsAliveCalibration::checkCorrectCalibrationType(){
  if(calibrationMode_=="PixelAlive")
    return true;
  else if(calibrationMode_=="unknown"){
    edm::LogInfo("SiPixelIsAliveCalibration") <<  "calibration mode is: " << calibrationMode_ << ", continuing anyway..." ;
    return true;
  }
  else{
    //    edm::LogError("SiPixelIsAliveCalibration")<< "unknown calibration mode for Pixel ALive, should be \"PixelAlive\" and is \"" << calibrationMode_ << "\"";
  }
  return false;

}
void SiPixelIsAliveCalibration::doSetup(const edm::ParameterSet &iConf){

}
void 
SiPixelIsAliveCalibration::calibrationSetup(const edm::EventSetup & iSetup){

}
bool
SiPixelIsAliveCalibration::doFits(uint32_t detid, std::vector<SiPixelCalibDigi>::const_iterator ipix){
  edm::LogInfo("SiPixelIsAliveCalibration") << "now looking at DetID " << detid << ", pixel " << ipix->row() << "," << ipix->col() << std::endl;
  
  double denom=0;
  double nom=0;
  for(uint32_t i=0; i<ipix->getnpoints(); i++){
    nom+=ipix->getnentries(i);
    denom+=calib_->getNTriggers();
    if(i>0)
      edm::LogWarning("SiPixelIsAliveCalibration::doFits") << " ERROR!!" << " number of vcal points is now " << i << " for detid " << detid << std::endl;
  }
  edm::LogInfo("SiPixelIsAliveCalibration") << "DetID/col/row " << detid << "/"<< ipix->col() << "/" << ipix->row() << ", now calculating efficiency: " << nom << "/" << denom <<"=" << nom/denom << std::endl;
  double eff = -1;
  if(denom>0)
    eff = nom/denom;
  if(bookkeeper_[detid]->getBinContent(ipix->col()+1,ipix->row()+1)==0){
    bookkeeper_[detid]->Fill(ipix->col(), ipix->row(), eff);
    summaries_[detid]->Fill(eff);
    float zerobin = summaries_[detid]->getBinContent(1);
    summaries_[detid]->setBinContent(1, zerobin-1);
  }
  else
    bookkeeper_[detid]->setBinContent(ipix->col()+1,ipix->row()+1,-2);
  return true;
}

void
SiPixelIsAliveCalibration::calibrationEnd(){
  // print summary of bad modules:
  for(std::map<uint32_t, MonitorElement *>::const_iterator idet= bookkeeper_.begin(); idet!=bookkeeper_.end();++idet){
    float idead = 0;
    float iunderthres=0;
    float imultiplefill=0;
    float itot=0;
    uint32_t detid=idet->first;

    setDQMDirectory(detid);
    for(int icol=1; icol <= bookkeeper_[detid]->getNbinsX(); ++icol){
      for(int irow=1; irow <= bookkeeper_[detid]->getNbinsY(); ++irow){
	itot++;
	double val = bookkeeper_[detid]->getBinContent(icol,irow);
	if(val==0)
	  idead++;
	if(val<mineff_)
	  iunderthres++;
	if(val==-2)
	  imultiplefill++;
	
      }
    }
    edm::LogInfo("SiPixelIsAliveCalibration") << "summary for " << translateDetIdToString(detid) << "\tfrac dead:" << idead/itot << " frac below " << mineff_ << ":" << iunderthres/itot << " bad " <<  imultiplefill/itot << std::endl;
  }

}
// ---------- method called to for each event  ------------

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelIsAliveCalibration);
