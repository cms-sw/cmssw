// -*- C++ -*-
//
// Package:    SiPixelGainCalibrationAnalysis
// Class:      SiPixelGainCalibrationAnalysis
// 
/**\class SiPixelGainCalibrationAnalysis SiPixelGainCalibrationAnalysis.cc CalibTracker/SiPixelGainCalibrationAnalysis/src/SiPixelGainCalibrationAnalysis.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Freya Blekman
//         Created:  Wed Nov 14 15:02:06 CET 2007
// $Id: SiPixelGainCalibrationAnalysis.cc,v 1.20 2008/04/21 12:39:04 fblekman Exp $
//
//

// user include files
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "SiPixelGainCalibrationAnalysis.h"
#include <sstream>
#include <math.h>
#include "TMath.h"
//
// constructors and destructor
//
SiPixelGainCalibrationAnalysis::SiPixelGainCalibrationAnalysis(const edm::ParameterSet& iConfig):
  SiPixelOfflineCalibAnalysisBase(iConfig),
  conf_(iConfig),
  bookkeeper_(),
  bookkeeper_pixels_(),
  nfitparameters_(iConfig.getUntrackedParameter<int>("numberOfFitParameters",2)),
  fitfunction_(iConfig.getUntrackedParameter<std::string>("fitFunctionRootFormula","pol1")),
  reject_plateaupoints_(iConfig.getUntrackedParameter<bool>("suppressPlateauInFit",true)),
  reject_single_entries_(iConfig.getUntrackedParameter<bool>("suppressPointsWithOneEntryOrLess",true)),
  reject_badpoints_frac_(iConfig.getUntrackedParameter<double>("suppressZeroAndPlateausInFitFrac",0)),
  bookBIGCalibPayload_(iConfig.getUntrackedParameter<bool>("saveFullPayloads",false)),
  savePixelHists_(iConfig.getUntrackedParameter<bool>("savePixelLevelHists",false)),
  chi2Threshold_(iConfig.getUntrackedParameter<double>("minChi2NDFforHistSave",10)),
  chi2ProbThreshold_(iConfig.getUntrackedParameter<double>("minChi2ProbforHistSave",0.05)),
  maxGainInHist_(iConfig.getUntrackedParameter<double>("maxGainInHist",10)),
  maxChi2InHist_(iConfig.getUntrackedParameter<double>("maxChi2InHist",25)),
  saveALLHistograms_(iConfig.getUntrackedParameter<bool>("saveAllHistograms",false)),
  filldb_(iConfig.getUntrackedParameter<bool>("writeDatabase",false)),
  recordName_(conf_.getParameter<std::string>("record")),
  appendMode_(conf_.getUntrackedParameter<bool>("appendMode",true)),
  theGainCalibrationDbInput_(0),
  theGainCalibrationDbInputOffline_(0),
  theGainCalibrationDbInputHLT_(0),
  theGainCalibrationDbInputService_(iConfig),
  gainlow_(10.),gainhi_(0.),pedlow_(255.),pedhi_(0.)
{
  ::putenv("CORAL_AUTH_USER=me");
  ::putenv("CORAL_AUTH_PASSWORD=test");   
  edm::LogInfo("SiPixelGainCalibrationAnalysis") << "now using fit function " << fitfunction_ << ", which has " << nfitparameters_ << " free parameters. " << std::endl;
  func_= new TF1("func",fitfunction_.c_str());
}

SiPixelGainCalibrationAnalysis::~SiPixelGainCalibrationAnalysis()
{
}
// member functions
//
// ------------ method called once each job just before starting event loop  ------------

std::vector<float> SiPixelGainCalibrationAnalysis::CalculateAveragePerColumn(uint32_t detid, std::string label){
  std::vector<float> result;
  int ncols= bookkeeper_[detid][label]->getNbinsX();
  int nrows= bookkeeper_[detid][label]->getNbinsY();
  for(int icol=1; icol<=ncols; ++icol){
    float val=0;
    float ntimes =0;
    for(int irow=1; irow<=nrows; ++irow){
      val+= bookkeeper_[detid][label]->getBinContent(icol,irow);
      ntimes++;
    }
    val/= ntimes;
    result.push_back(val);
  }
  return result;
}

bool
SiPixelGainCalibrationAnalysis::checkCorrectCalibrationType()
{
  if(calibrationMode_=="GainCalibration")
    return true;
  else if(calibrationMode_=="unknown"){
    edm::LogInfo("SiPixelGainCalibrationAnalysis") <<  "calibration mode is: " << calibrationMode_ << ", continuing anyway..." ;
    return true;
  }
  else{
    //    edm::LogError("SiPixelGainCalibrationAnalysis") << "unknown calibration mode for Gain calibration, should be \"Gain\" and is \"" << calibrationMode_ << "\"";
  }
  return false;
}

void SiPixelGainCalibrationAnalysis::calibrationSetup(const edm::EventSetup&)
{
}
//------- summary printing method. Very verbose.
void
SiPixelGainCalibrationAnalysis::printSummary(){

  uint32_t detid=0;
  for(std::map<uint32_t,std::map<std::string,MonitorElement *> >::const_iterator idet = bookkeeper_.begin(); idet != bookkeeper_.end(); ++idet){
    if(detid==idet->first)
      continue;// only do things once per detid
    detid=idet->first;
    std::vector<float> gainvec=CalculateAveragePerColumn(detid,"gain_2d");
    std::vector<float> pedvec =CalculateAveragePerColumn(detid,"ped_2d");
    std::vector<float> chi2vec = CalculateAveragePerColumn(detid,"chi2_2d");
    std::ostringstream summarytext;

    summarytext << "Summary for det ID " << detid << "(" << translateDetIdToString(detid) << ")\n";
    summarytext << "\t Following: values per column: column #, gain, pedestal, chi2\n";
    for(uint32_t i=0; i<gainvec.size(); i++)
      summarytext << "\t " << i << " \t" << gainvec[i] << " \t" << pedvec[i] << " \t" << chi2vec[i] << "\n";
    summarytext << "\t list of pixels with high chi2 (chi2> " << chi2Threshold_ << "): \n";

    
    for(std::map<std::string, MonitorElement *>::const_iterator ipix = bookkeeper_pixels_[detid].begin(); ipix!=bookkeeper_pixels_[detid].end(); ++ipix)
      summarytext << "\t " << ipix->first << "\n";
    edm::LogInfo("SiPixelGainCalibrationAnalysis") << summarytext.str() << std::endl;

  }

}

// ------------ method called once each job just after ending the event loop  ------------

void 
SiPixelGainCalibrationAnalysis::calibrationEnd() {

  //  printSummary();
  
  // this is where we loop over all histograms and save the database objects
  if(filldb_)
    fillDatabase();
}
//-----------method to fill the database
void SiPixelGainCalibrationAnalysis::fillDatabase(){
  // only create when necessary.
  // process the minimum and maximum gain & ped values...


  if(gainlow_>gainhi_){  
    float temp=gainhi_;
    gainhi_=gainlow_;
    gainlow_=temp;
  }
  if(pedlow_>pedhi_){  
    float temp=pedhi_;
    pedhi_=pedlow_;
    pedlow_=temp;
  }
 
  theGainCalibrationDbInput_ = new SiPixelGainCalibration(pedlow_,pedhi_,gainlow_,gainhi_);
  theGainCalibrationDbInputHLT_ = new SiPixelGainCalibrationForHLT(pedlow_,pedhi_,gainlow_,gainhi_);
  theGainCalibrationDbInputOffline_ = new SiPixelGainCalibrationOffline(pedlow_,pedhi_,gainlow_,gainhi_);

  uint32_t nchannels=0;
  uint32_t nmodules=0;
  //  std::cout << "now starting loop on detids" << std::endl;
  uint32_t detid=0;
  for(std::map<uint32_t,std::map<std::string, MonitorElement *> >::const_iterator idet=bookkeeper_.begin(); idet!= bookkeeper_.end(); ++idet){
    if(detid==idet->first)
      continue;
    detid=idet->first;
    edm::LogInfo("SiPixelGainCalibrationAnalysis") << "now creating database object for detid " << detid << std::endl;
    // Get the module sizes.

    size_t nrows = bookkeeper_[detid]["gain_2d"]->getNbinsY();
    size_t ncols = bookkeeper_[detid]["gain_2d"]->getNbinsX();   
    size_t nrowsrocsplit = theGainCalibrationDbInputHLT_->getNumberOfRowsToAverageOver();
    if(theGainCalibrationDbInputOffline_->getNumberOfRowsToAverageOver()!=nrowsrocsplit)
      throw  cms::Exception("GainCalibration Payload configuration error")
	<< "[SiPixelGainCalibrationAnalysis::fillDatabase] ERROR the SiPixelGainCalibrationOffline and SiPixelGainCalibrationForHLT database payloads have different settings for the number of rows per roc: " << theGainCalibrationDbInputHLT_->getNumberOfRowsToAverageOver() << "(HLT), " << theGainCalibrationDbInputOffline_->getNumberOfRowsToAverageOver() << "(offline)";
    std::vector<char> theSiPixelGainCalibrationPerPixel;
    std::vector<char> theSiPixelGainCalibrationPerColumn;
    std::vector<char> theSiPixelGainCalibrationGainPerColPedPerPixel;
    
    // Loop over columns and rows of this DetID
    //    std::cout <<" now starting loop over pixels..." << std::endl;
    
    for(size_t i=1; i<=ncols; i++) {
      float pedforthiscol[2]={0,0};
      float gainforthiscol[2]={0,0};
      int nusedrows[2]={0,0};
      //      std::cout << "now lookign at col " << i << std::endl;
      for(size_t j=1; j<=nrows; j++) {
	nchannels++;
	int iglobalrow=0;
	if(nrows>nrowsrocsplit)
	  iglobalrow=1;
	float ped = bookkeeper_[detid]["ped_2d"]->getBinContent(i,j);
	float gain = bookkeeper_[detid]["gain_2d"]->getBinContent(i,j);
	
	//	std::cout << "detid: "<< detid << ", looking at pixel row,col " << j << ","<<i << " gain,ped=" <<gain << "," << ped << std::endl;
	if(ped==0 && gain==0){// dead pixel
	  //	  std::cout << "dead!" << std::endl;
	  theGainCalibrationDbInput_->setDeadPixel(theSiPixelGainCalibrationPerPixel);
	  theGainCalibrationDbInputOffline_->setDeadPixel(theSiPixelGainCalibrationGainPerColPedPerPixel);
	}
	else{// pixel not dead
	  theGainCalibrationDbInput_->setData(ped,gain,theSiPixelGainCalibrationPerPixel);
	  theGainCalibrationDbInputOffline_->setDataPedestal(ped, theSiPixelGainCalibrationGainPerColPedPerPixel);
	
	//	std::cout <<"done with database filling..." << std::endl;

	  pedforthiscol[iglobalrow]+=ped;
	  gainforthiscol[iglobalrow]+=gain;
	  nusedrows[iglobalrow]++;
	}
	if(j%nrowsrocsplit==nrowsrocsplit){  
	  if(nusedrows[iglobalrow]>0){// good column
	    pedforthiscol[iglobalrow]/=(float)nusedrows[iglobalrow];
	    gainforthiscol[iglobalrow]/=(float)nusedrows[iglobalrow];
	    //	    std::cout << "good column ave gain,ped " << gainforthiscol[iglobalrow] << "," <<  pedforthiscol[iglobalrow] << std::endl;
	    theGainCalibrationDbInputOffline_->setDataGain(gainforthiscol[iglobalrow],nrowsrocsplit,theSiPixelGainCalibrationGainPerColPedPerPixel);
	    theGainCalibrationDbInputHLT_->setData(pedforthiscol[iglobalrow],gainforthiscol[iglobalrow],theSiPixelGainCalibrationPerColumn);
	  }
	  else if(nusedrows[iglobalrow]=0){// dead column!
	    //	    std::cout << "dead column!" << std::endl;
	    theGainCalibrationDbInputOffline_->setDeadColumn(nrowsrocsplit,theSiPixelGainCalibrationGainPerColPedPerPixel);
	    theGainCalibrationDbInputHLT_->setDeadColumn(nrowsrocsplit,theSiPixelGainCalibrationPerColumn);
	  }
	}
      }
    }

    //    std::cout << "setting range..." << std::endl;
    SiPixelGainCalibration::Range range(theSiPixelGainCalibrationPerPixel.begin(),theSiPixelGainCalibrationPerPixel.end());
    SiPixelGainCalibrationForHLT::Range hltrange(theSiPixelGainCalibrationPerColumn.begin(),theSiPixelGainCalibrationPerColumn.end());
    SiPixelGainCalibrationOffline::Range offlinerange(theSiPixelGainCalibrationGainPerColPedPerPixel.begin(),theSiPixelGainCalibrationGainPerColPedPerPixel.end());
    
    //    std::cout <<"putting things in db..." << std::endl;
    // now start creating the various database objects
    if( bookBIGCalibPayload_)
      if( !theGainCalibrationDbInput_->put(detid,range,ncols) )
	edm::LogError("SiPixelGainCalibrationAnalysis")<<"warning: detid already exists for Pixel-level calibration database"<<std::endl;
    if( !theGainCalibrationDbInputOffline_->put(detid,offlinerange,ncols) )
      edm::LogError("SiPixelGainCalibrationAnalysis")<<"warning: detid already exists for Offline (gain per col, ped per pixel) calibration database"<<std::endl;
    if(!theGainCalibrationDbInputHLT_->put(detid,hltrange, ncols) )
      edm::LogError("SiPixelGainCalibrationAnalysis")<<"warning: detid already exists for HLT (pedestal and gain per column) calibration database"<<std::endl;
  }
  
  edm::LogInfo("SiPixelGainCalibrationAnalysis") << " ---> PIXEL Modules  " << nmodules  << "\n"
						 << " ---> PIXEL Channels " << nchannels << std::endl;

  edm::LogInfo(" --- writing to DB!");
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if(!mydbservice.isAvailable() ){
    edm::LogError("db service unavailable");
    return;
    if( mydbservice->isNewTagRequest(recordName_) ){
      if( bookBIGCalibPayload_)
	mydbservice->createNewIOV<SiPixelGainCalibration>(
							  theGainCalibrationDbInput_, 
							  mydbservice->beginOfTime(),
							  mydbservice->endOfTime(),
							  recordName_);
      
      mydbservice->createNewIOV<SiPixelGainCalibrationForHLT>(
							   theGainCalibrationDbInputHLT_,
							   mydbservice->beginOfTime(),
							   mydbservice->endOfTime(),
							   recordName_);
      
      mydbservice->createNewIOV<SiPixelGainCalibrationOffline>(
							       theGainCalibrationDbInputOffline_,
							       mydbservice->beginOfTime(),
							       mydbservice->endOfTime(),
							       recordName_);
      
    } 
    else {
      if( bookBIGCalibPayload_)
	mydbservice->appendSinceTime<SiPixelGainCalibration>(
							     theGainCalibrationDbInput_, 
							     mydbservice->currentTime(),
							     recordName_);
      
      mydbservice->appendSinceTime<SiPixelGainCalibrationForHLT>(
								 theGainCalibrationDbInputHLT_, 
								 mydbservice->currentTime(),
								 recordName_);
      
      mydbservice->appendSinceTime<SiPixelGainCalibrationOffline>(
								  theGainCalibrationDbInputOffline_, 
								  mydbservice->currentTime(),
								  recordName_);
    }
    edm::LogInfo(" --- all OK");
  } 
}
// ------------ method called to do fits to all objects available  ------------
bool
SiPixelGainCalibrationAnalysis::doFits(uint32_t detid, std::vector<SiPixelCalibDigi>::const_iterator ipix)
{
  bool makehistopersistent = saveALLHistograms_;
  // first, fill the input arrays to the TLinearFitter.
  double xvals[201];
  double yvals[200];
  double yerrvals[200];
  double xvalsall[201];
  float  xvalsasfloatsforDQM[201];
  double yvalsall[200];
  double yerrvalsall[200];
  int npoints=0;
  int nallpoints=0;
  bool use_point=true;
  for(uint32_t ii=0; ii< ipix->getnpoints() && ii<200; ii++){
    nallpoints++;
    use_point=true;
    xvalsasfloatsforDQM[ii]=xvalsall[ii]=vCalValues_[ii];
    yerrvalsall[ii]=yvalsall[ii]=0;
    if(ipix->getnentries(ii)>0){
      yvalsall[ii]=ipix->getsum(ii)/(float)ipix->getnentries(ii);
      yerrvalsall[ii]=ipix->getsumsquares(ii)/(float)ipix->getnentries(ii);
      yerrvalsall[ii]-=pow(yvalsall[ii],2);
      yerrvalsall[ii]=sqrt(yerrvalsall[ii]);
    }
  }
  
  // calculate plateau value from last 3 full entries
  double plateauval=0;
  for(int ii=nallpoints-1; ii>=0 && npoints<3; --ii){
    if(yvalsall[ii]>0 && yerrvalsall[ii]>0){
      plateauval+=yvalsall[ii];
      npoints++;
    }
  }
  plateauval/=npoints;
  double maxgoodvalinfit=plateauval*(1.-reject_badpoints_frac_);
  if(maxgoodvalinfit<1)
    maxgoodvalinfit=255*(1.-reject_badpoints_frac_);
  npoints=0;
  for(int ii=0; ii<nallpoints; ++ii){
    // now selecting the appropriate points for the fit.
    use_point=true;
    if(ipix->getnentries(ii)<=1 && reject_single_entries_)
      use_point=false;
    if(ipix->getnentries(ii)==0 && reject_badpoints_)
      use_point=false;
    if(yvalsall[ii]>maxgoodvalinfit)
      use_point=false;
    
    if(use_point){
      xvals[npoints]=xvalsall[ii];
      yvals[npoints]=yvalsall[ii];
      yerrvals[npoints]=yerrvalsall[ii];
      npoints++;
    }
  }
  int result=1;
  float chi2,slope,intercept,prob;
  prob=chi2=-1;
  slope=intercept=0;
  TLinearFitter fitter(nfitparameters_,fitfunction_.c_str());
 
  if(npoints>=2){
    fitter.AssignData(npoints,1,xvals,yvals,yerrvals);
    
    // and do the fit:
    result = fitter.Eval();
    
    slope = fitter.GetParameter(1);
    intercept = fitter.GetParameter(0);
    for(int i=0; i< func_->GetNpar();i++)
      func_->SetParameter(i,fitter.GetParameter(i));
    
    // convert the gain and pedestal parameters to functional form y= x/gain+ ped
    if(slope!=0)
      slope = 1./ slope;
    
    chi2 = fitter.GetChisquare()/fitter.GetNumberFreeParameters();
    prob = TMath::Prob(fitter.GetChisquare(),fitter.GetNumberFreeParameters());
    if(slope<0)
      makehistopersistent=true;
    if(chi2>chi2Threshold_ && chi2Threshold_>=0)
      makehistopersistent=true;
    if(prob<chi2ProbThreshold_)
      makehistopersistent=true;
    if(result==1)
      makehistopersistent=true;

    if(result==0){
      if(slope<gainlow_)
	gainlow_=slope;
      if(slope>gainhi_)
	gainhi_=slope;
      if(intercept>pedhi_)
	pedhi_=intercept;
      if(intercept<pedlow_)
	pedlow_=intercept;
      bookkeeper_[detid]["gain_1d"]->Fill(slope);
      bookkeeper_[detid]["gain_2d"]->Fill(ipix->col(),ipix->row(),slope);
      bookkeeper_[detid]["ped_1d"]->Fill(intercept);
      bookkeeper_[detid]["ped_2d"]->Fill(ipix->col(),ipix->row(),intercept);
      bookkeeper_[detid]["chi2_1d"]->Fill(chi2);
      bookkeeper_[detid]["chi2_2d"]->Fill(ipix->col(),ipix->row(),chi2);
      bookkeeper_[detid]["prob_1d"]->Fill(prob);
      bookkeeper_[detid]["prob_2d"]->Fill(ipix->col(),ipix->row(),prob);
    }
  }
  
  if(!savePixelHists_)
    return true;
  if(makehistopersistent){
    setDQMDirectory(detid);
    std::ostringstream pixelinfo;
    pixelinfo << "GainCurve_row_" << ipix->row() << "_col_" << ipix->col();
    std::string tempname=translateDetIdToString(detid);
    tempname+="_";
    tempname+=pixelinfo.str();
    // and book the histo
    // fill the last value of the vcal array...
    xvalsasfloatsforDQM[nallpoints]=256;
    if(nallpoints>2)
      xvalsasfloatsforDQM[nallpoints]=2*xvalsasfloatsforDQM[nallpoints-1]-xvalsasfloatsforDQM[nallpoints-2];
    bookkeeper_pixels_[detid][pixelinfo.str()] =  bookDQMHistogram1D(detid,pixelinfo.str(),tempname,nallpoints,xvalsasfloatsforDQM);
    edm::LogInfo("SiPixelGainCalibrationAnalysis") << "now saving histogram for pixel " << tempname << ", gain = " << slope << ", pedestal = " << intercept << ", chi2/NDF=" << chi2 << "(prob:" << prob << "), fit status " << result;
    for(int ii=0; ii<nallpoints; ++ii){
      bookkeeper_pixels_[detid][pixelinfo.str()]->setBinContent(ii+1,yvalsall[ii]);
      bookkeeper_pixels_[detid][pixelinfo.str()]->setBinError(ii+1,yerrvalsall[ii]);
    }
    
    addTF1ToDQMMonitoringElement(bookkeeper_pixels_[detid][pixelinfo.str()],func_);
  } 
  return true;
}
// ------------ method called to do fill new detids  ------------
void 
SiPixelGainCalibrationAnalysis::newDetID(uint32_t detid)
{
  setDQMDirectory(detid);
  std::string tempname=translateDetIdToString(detid);
  bookkeeper_[detid]["gain_1d"] = bookDQMHistogram1D(detid,"Gain1d","gain for "+tempname,100,0.,maxGainInHist_);
  bookkeeper_[detid]["gain_2d"] = bookDQMHistoPlaquetteSummary2D(detid, "Gain2d","gain for "+tempname);
  bookkeeper_[detid]["ped_1d"] = bookDQMHistogram1D(detid,"Pedestal1d","pedestal for "+tempname,256,0.,256.);
  bookkeeper_[detid]["ped_2d"] = bookDQMHistoPlaquetteSummary2D(detid,"Pedestal2d","pedestal for "+tempname);
  bookkeeper_[detid]["chi2_1d"] = bookDQMHistogram1D(detid,"GainChi2NDF1d","#chi^{2}/NDOF for "+tempname,100,0.,maxChi2InHist_);
  bookkeeper_[detid]["chi2_2d"] = bookDQMHistoPlaquetteSummary2D(detid,"GainChi2NDF2d","#chi^{2}/NDOF for "+tempname);
  bookkeeper_[detid]["prob_1d"] = bookDQMHistogram1D(detid,"GainChi2Prob1d","P(#chi^{2},NDOF) for "+tempname,100,0.,1.0);
  bookkeeper_[detid]["prob_2d"] = bookDQMHistoPlaquetteSummary2D(detid,"GainChi2Prob2d","P(#chi^{2},NDOF) for "+tempname);

}
//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelGainCalibrationAnalysis);
