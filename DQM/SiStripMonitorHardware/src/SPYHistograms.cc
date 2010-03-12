#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/SiStripMonitorHardware/interface/SiStripFEDSpyBuffer.h"
#include "DQM/SiStripMonitorHardware/interface/SPYHistograms.h"

using edm::LogError;
using edm::LogWarning;
using edm::LogInfo;


SPYHistograms::SPYHistograms()
{
  dqm_ = 0;
}

SPYHistograms::~SPYHistograms()
{
}
 

void 
SPYHistograms::initialise(const edm::ParameterSet& iConfig,
			  std::ostringstream* pDebugStream){

  getConfigForHistogram("nNoData",iConfig,pDebugStream);
  getConfigForHistogram("nLowRange",iConfig,pDebugStream);
  getConfigForHistogram("nHighRange",iConfig,pDebugStream);
  getConfigForHistogram("nMinZero",iConfig,pDebugStream);
  getConfigForHistogram("nMaxSat",iConfig,pDebugStream);
  getConfigForHistogram("nLowPb",iConfig,pDebugStream);
  getConfigForHistogram("nHighPb",iConfig,pDebugStream);
  getConfigForHistogram("nOutOfSync",iConfig,pDebugStream);
  getConfigForHistogram("nOtherPbs",iConfig,pDebugStream);
  getConfigForHistogram("nApvErrorBit",iConfig,pDebugStream);
  getConfigForHistogram("nApvAddressError",iConfig,pDebugStream);
  getConfigForHistogram("nNegativePeds",iConfig,pDebugStream);

  getConfigForHistogram("nNoDatavsTime",iConfig,pDebugStream);
  getConfigForHistogram("nLowRangevsTime",iConfig,pDebugStream);
  getConfigForHistogram("nHighRangevsTime",iConfig,pDebugStream);
  getConfigForHistogram("nMinZerovsTime",iConfig,pDebugStream);
  getConfigForHistogram("nMaxSatvsTime",iConfig,pDebugStream);
  getConfigForHistogram("nLowPbvsTime",iConfig,pDebugStream);
  getConfigForHistogram("nHighPbvsTime",iConfig,pDebugStream);
  getConfigForHistogram("nOutOfSyncvsTime",iConfig,pDebugStream);
  getConfigForHistogram("nOtherPbsvsTime",iConfig,pDebugStream);
  getConfigForHistogram("nApvErrorBitvsTime",iConfig,pDebugStream);
  getConfigForHistogram("nApvAddressErrorvsTime",iConfig,pDebugStream);
  getConfigForHistogram("nNegativePedsvsTime",iConfig,pDebugStream);

  getConfigForHistogram("meanDigitalLowvsTime",iConfig,pDebugStream);


  getConfigForHistogram("NoData",iConfig,pDebugStream);
  getConfigForHistogram("LowRange",iConfig,pDebugStream);
  getConfigForHistogram("HighRange",iConfig,pDebugStream);
  getConfigForHistogram("MinZero",iConfig,pDebugStream);
  getConfigForHistogram("MaxSat",iConfig,pDebugStream);
  getConfigForHistogram("LowPb",iConfig,pDebugStream);
  getConfigForHistogram("HighPb",iConfig,pDebugStream);
  getConfigForHistogram("OutOfSync",iConfig,pDebugStream);
  getConfigForHistogram("OtherPbs",iConfig,pDebugStream);
  getConfigForHistogram("ApvErrorBit",iConfig,pDebugStream);
  getConfigForHistogram("ApvAddressError",iConfig,pDebugStream);
  getConfigForHistogram("NegativePeds",iConfig,pDebugStream);

  getConfigForHistogram("FrameRange",iConfig,pDebugStream);
  getConfigForHistogram("FrameMin",iConfig,pDebugStream);
  getConfigForHistogram("FrameMax",iConfig,pDebugStream);
  getConfigForHistogram("Baseline",iConfig,pDebugStream);

  getConfigForHistogram("NoDataDetailed",iConfig,pDebugStream);
  getConfigForHistogram("LowRangeDetailed",iConfig,pDebugStream);
  getConfigForHistogram("HighRangeDetailed",iConfig,pDebugStream);
  getConfigForHistogram("MinZeroDetailed",iConfig,pDebugStream);
  getConfigForHistogram("MaxSatDetailed",iConfig,pDebugStream);
  getConfigForHistogram("LowPbDetailed",iConfig,pDebugStream);
  getConfigForHistogram("HighPbDetailed",iConfig,pDebugStream);
  getConfigForHistogram("OutOfSyncDetailed",iConfig,pDebugStream);
  getConfigForHistogram("OtherPbsDetailed",iConfig,pDebugStream);
  getConfigForHistogram("ApvErrorBitDetailed",iConfig,pDebugStream);
  getConfigForHistogram("ApvAddressErrorDetailed",iConfig,pDebugStream);
  getConfigForHistogram("NegativePedsDetailed",iConfig,pDebugStream);

  getConfigForHistogram("PositionOfFirstHeaderBitDetailed",iConfig,pDebugStream);
  getConfigForHistogram("PositionOfFirstTrailerBitDetailed",iConfig,pDebugStream);
  getConfigForHistogram("DistanceHeaderTrailerDetailed",iConfig,pDebugStream);

}

void SPYHistograms::fillGainHistograms(const Trends & aTrendElement, const double aTime) {

  fillHistogram(meanDigitalLowvsTime_,aTime,aTrendElement.meanDigitalLow);

}

void SPYHistograms::fillCountersHistograms(const ErrorCounters & aCounter, const double aTime) {


  fillHistogram(nNoData_,aCounter.nNoData);
  fillHistogram(nLowRange_,aCounter.nLowRange);
  fillHistogram(nHighRange_,aCounter.nHighRange);
  fillHistogram(nMinZero_,aCounter.nMinZero);
  fillHistogram(nMaxSat_,aCounter.nMaxSat);
  fillHistogram(nLowPb_,aCounter.nLowPb);
  fillHistogram(nHighPb_,aCounter.nHighPb);
  fillHistogram(nOutOfSync_,aCounter.nOOS);
  fillHistogram(nOtherPbs_,aCounter.nOtherPbs);
  fillHistogram(nApvErrorBit_,aCounter.nAPVError);
  fillHistogram(nApvAddressError_,aCounter.nAPVAddressError);
  fillHistogram(nNegativePeds_,aCounter.nNegPeds);

  fillHistogram(nNoDatavsTime_,aTime,aCounter.nNoData);
  fillHistogram(nLowRangevsTime_,aTime,aCounter.nLowRange);
  fillHistogram(nHighRangevsTime_,aTime,aCounter.nHighRange);
  fillHistogram(nMinZerovsTime_,aTime,aCounter.nMinZero);
  fillHistogram(nMaxSatvsTime_,aTime,aCounter.nMaxSat);
  fillHistogram(nLowPbvsTime_,aTime,aCounter.nLowPb);
  fillHistogram(nHighPbvsTime_,aTime,aCounter.nHighPb);
  fillHistogram(nOutOfSyncvsTime_,aTime,aCounter.nOOS);
  fillHistogram(nOtherPbsvsTime_,aTime,aCounter.nOtherPbs);
  fillHistogram(nApvErrorBitvsTime_,aTime,aCounter.nAPVError);
  fillHistogram(nApvAddressErrorvsTime_,aTime,aCounter.nAPVAddressError);
  fillHistogram(nNegativePedsvsTime_,aTime,aCounter.nNegPeds);


}


void SPYHistograms::fillFEDHistograms(const Errors & aErr,const unsigned int aFedId) {
  //else is used when filling the Error struct per channel, to avoid redundancy
  //use only if here, in case channels in the same FED have different 
  //"exclusive" error types...

  if (aErr.hasNoData) {
    //fill in channel loop with delay FPGA number...
    //fillHistogram(noData_,aFedId);
  }
  if (aErr.hasLowRange) { 
    fillHistogram(lowRange_,aFedId);
  }
  if (aErr.hasHighRange) { 
    fillHistogram(highRange_,aFedId);
  }
  if (aErr.hasMinZero) {
    fillHistogram(minZero_,aFedId);
  }
  if (aErr.hasMaxSat) {
    fillHistogram(maxSat_,aFedId);
  }
  if (aErr.hasLowPb) {
    fillHistogram(lowPb_,aFedId);
  }
  if (aErr.hasHighPb) {
    fillHistogram(highPb_,aFedId);
  }
  if (aErr.hasOOS) {
    fillHistogram(outOfSync_,aFedId);
  }
  if (aErr.hasOtherPbs) {
    fillHistogram(otherPbs_,aFedId);
  }
  if (aErr.hasErrorBit0 || aErr.hasErrorBit1) {
    fillHistogram(apvErrorBit_,aFedId);
  }
  if (aErr.hasAPVAddressError0 || aErr.hasAPVAddressError1) {
    fillHistogram(apvAddressError_,aFedId);
  }
  if (aErr.hasNegPeds) {
    fillHistogram(negativePeds_,aFedId);
  }


}


void SPYHistograms::fillDetailedHistograms(const Errors & aErr,
					   const sistrip::SpyUtilities::Frame & aFrame, 
					   const unsigned int aFedId,
					   const unsigned int aFedChannel) {

  bookFEDHistograms(aFedId,aErr);

  if (!aErr.hasNoData) {
    fillHistogram(frameRange_,sistrip::SpyUtilities::range(aFrame));
    fillHistogram(frameMin_,aFrame.digitalLow);
    fillHistogram(frameMax_,aFrame.digitalHigh);
    fillHistogram(baseline_,aFrame.baseline);
  }

  if (aErr.hasNoData) {
    fillHistogram(noDataDetailed_[aFedId],aFedChannel);
    //fillHistogram(noData_,24*aFedId+static_cast<unsigned int>(aFedChannel/4.));
    fillHistogram(noData_,static_cast<unsigned int>(aFedChannel/4.));
  }

  if (aErr.hasLowRange) { 
    fillHistogram(lowRangeDetailed_[aFedId],aFedChannel);
  }
  if (aErr.hasHighRange) { 
    fillHistogram(highRangeDetailed_[aFedId],aFedChannel);
  }
  if (aErr.hasMinZero) {
    fillHistogram(minZeroDetailed_[aFedId],aFedChannel);
  }
  if (aErr.hasMaxSat) {
    fillHistogram(maxSatDetailed_[aFedId],aFedChannel);
  }
  if (aErr.hasLowPb) {
    fillHistogram(lowPbDetailed_[aFedId],aFedChannel);
  }
  if (aErr.hasHighPb) {
    fillHistogram(highPbDetailed_[aFedId],aFedChannel);
  }
  if (aErr.hasOOS) {
    fillHistogram(outOfSyncDetailed_[aFedId],aFedChannel);
  }
  if (aErr.hasOtherPbs) {
    fillHistogram(otherPbsDetailed_[aFedId],aFedChannel);
  }
  if (aErr.hasErrorBit0) {
    fillHistogram(apvErrorBitDetailed_[aFedId],2*aFedChannel);
  }
  if (aErr.hasErrorBit1) {
    fillHistogram(apvErrorBitDetailed_[aFedId],2*aFedChannel+1);
  }
  if (aErr.hasAPVAddressError0) {
    fillHistogram(apvAddressErrorDetailed_[aFedId],2*aFedChannel);
  }
  if (aErr.hasAPVAddressError1) {
    fillHistogram(apvAddressErrorDetailed_[aFedId],2*aFedChannel+1);
  }
  if (aErr.hasNegPeds) {
    fillHistogram(negativePedsDetailed_[aFedId],aFedChannel);
  }

  fillHistogram(positionOfFirstHeaderBitDetailed_[aFedId],aFrame.firstHeaderBit);
  fillHistogram(positionOfFirstTrailerBitDetailed_[aFedId],aFrame.firstTrailerBit);
  if (aFrame.firstTrailerBit != sistrip::SPY_SAMPLES_PER_CHANNEL &&
      aFrame.firstHeaderBit != sistrip::SPY_SAMPLES_PER_CHANNEL)
    fillHistogram(distanceHeaderTrailerDetailed_[aFedId],aFrame.firstTrailerBit-aFrame.firstHeaderBit);
  else fillHistogram(distanceHeaderTrailerDetailed_[aFedId],0);
}


void SPYHistograms::bookTopLevelHistograms(DQMStore* dqm)
{

  dqm_ = dqm;
  //get FED IDs
  const unsigned int siStripFedIdMin = sistrip::FED_ID_MIN;
  const unsigned int siStripFedIdMax = sistrip::FED_ID_MAX;

  //book FED level histograms: resize boolean so 
  //they will be booked only once for the first event.
  histosBooked_.resize(siStripFedIdMax+1,false);


  for (std::map<std::string,HistogramConfig>::iterator iC = histogramConfig_.begin(); iC != histogramConfig_.end(); iC++){

    LogDebug("SPYHistograms") << " -- Config name : " << iC->first << ", isEnabled = " << iC->second.enabled << std::endl;
  }


  //book histos
  nNoData_ = bookHistogram("nNoData","nNoData",
			   ";n_{noData};n_{entries}",
			   "n_{noData}");

  nLowRange_ = bookHistogram("nLowRange","nLowRange",
			     ";n_{LowRange};n_{entries}",
			     "n_{LowRange}");

  nHighRange_ = bookHistogram("nHighRange","nHighRange",
			      ";n_{HighRange};n_{entries}",
			      "n_{HighRange}");

  nMinZero_ = bookHistogram("nMinZero","nMinZero",
			    ";n_{MinZero};n_{entries}",
			    "n_{MinZero}");

  nMaxSat_ = bookHistogram("nMaxSat","nMaxSat",
			   ";n_{MaxSat};n_{entries}",
			   "n_{MaxSat}");
  
  nLowPb_ = bookHistogram("nLowPb","nLowPb",
			       ";n_{lowPb};n_{entries}",
			       "n_{lowPb}");

  nHighPb_ = bookHistogram("nHighPb","nHighPb",
				";n_{highPb};n_{entries}",
				"n_{highPb}");
  
  nOutOfSync_ = bookHistogram("nOutOfSync","nOutOfSync",
			      ";n_{OOS};n_{entries}",
			      "n_{OOS}");

  nOtherPbs_ = bookHistogram("nOtherPbs","nOtherPbs",
			     ";n_{OtherPbs};n_{entries}",
			     "n_{OtherPbs}");

  nApvErrorBit_ = bookHistogram("nApvErrorBit","nApvErrorBit",
				";n_{APVerror};n_{entries}",
				"n_{APVerror}"
				);

  nApvAddressError_ = bookHistogram("nApvAddressError","nApvAddressError",
				    ";n_{APVAddressError};n_{entries}",
				    "n_{APVAddressError}"
				    );

  nNegativePeds_ = bookHistogram("nNegativePeds","nNegativePeds",
				 ";n_{negPeds};n_{entries}",
				 "n_{negPeds}"
				 );
  


  noData_ = bookHistogram("NoData","NoData",
			  ";delayFPGA;n_{noData}",
			  //24*(siStripFedIdMax+1)-24*siStripFedIdMin,
			  //24*siStripFedIdMin,24*(siStripFedIdMax+1),
			  24,0,24,
			  "delayFPGA index"
			  );

  lowRange_ = bookHistogram("LowRange","LowRange",
			     ";fedId;n_{LowRange}",
			     siStripFedIdMax-siStripFedIdMin+1,
			     siStripFedIdMin-0.5,siStripFedIdMax+0.5,
			     "FED-ID"
			     );
  
  highRange_ = bookHistogram("HighRange","HighRange",
			     ";fedId;n_{HighRange}",
			     siStripFedIdMax-siStripFedIdMin+1,
			     siStripFedIdMin-0.5,siStripFedIdMax+0.5,
			     "FED-ID"
			     );

  minZero_ = bookHistogram("MinZero","MinZero",
			   ";fedId;n_{MinZero}",
			   siStripFedIdMax-siStripFedIdMin+1,
			   siStripFedIdMin-0.5,siStripFedIdMax+0.5,
			   "FED-ID"
			   );

  maxSat_ = bookHistogram("MaxSat","MaxSat",
			  ";fedId;n_{MaxSat}",
			  siStripFedIdMax-siStripFedIdMin+1,
			  siStripFedIdMin-0.5,siStripFedIdMax+0.5,
			  "FED-ID"
			  );
  
  lowPb_ = bookHistogram("LowPb","LowPb",
			       ";fedId;n_{lowPb}",
			      siStripFedIdMax-siStripFedIdMin+1,
			      siStripFedIdMin-0.5,siStripFedIdMax+0.5,
			      "FED-ID"
			      );

  highPb_ = bookHistogram("HighPb","HighPb",
				";fedId;n_{highPb}",
			       siStripFedIdMax-siStripFedIdMin+1,
			       siStripFedIdMin-0.5,siStripFedIdMax+0.5,
			       "FED-ID"
			       );
  
  outOfSync_ = bookHistogram("OutOfSync","OutOfSync",
			      ";fedId;n_{OOS}",
			     siStripFedIdMax-siStripFedIdMin+1,
			     siStripFedIdMin-0.5,siStripFedIdMax+0.5,
			     "FED-ID"
			     );

  otherPbs_ = bookHistogram("OtherPbs","OtherPbs",
			     ";fedId;n_{OtherPbs}",
			    siStripFedIdMax-siStripFedIdMin+1,
			    siStripFedIdMin-0.5,siStripFedIdMax+0.5,
			    "FED-ID"
			    );

  apvErrorBit_ = bookHistogram("ApvErrorBit","ApvErrorBit",
				";fedId;n_{APVerror}",
			       siStripFedIdMax-siStripFedIdMin+1,
			       siStripFedIdMin-0.5,siStripFedIdMax+0.5,
			       "FED-ID"
			       );

  apvAddressError_ = bookHistogram("ApvAddressError","ApvAddressError",
				    ";fedId;n_{APVAddressError}",
				   siStripFedIdMax-siStripFedIdMin+1,
				   siStripFedIdMin-0.5,siStripFedIdMax+0.5,
				   "FED-ID"
				   );

  negativePeds_ = bookHistogram("NegativePeds","NegativePeds",
				 ";fedId;n_{negPeds}",
				siStripFedIdMax-siStripFedIdMin+1,
				siStripFedIdMin-0.5,siStripFedIdMax+0.5,
				"FED-ID"
				);
  
  frameRange_ = bookHistogram("FrameRange","FrameRange",
			      ";range;n_{entries}",
			      1024,0,1024,
			      "range (adc counts)");

  frameMin_ = bookHistogram("FrameMin","FrameMin",
			    ";Zero-light level;n_{entries}",
			    800,0,800,
			    "Zero-light level (adc counts)");

  frameMax_ = bookHistogram("FrameMax","FrameMax",
			    ";Tick height;n_{entries}",
			    824,200,1024,
			    "Tick height (adc counts)");

  baseline_ = bookHistogram("Baseline","Baseline",
			    ";Baseline;n_{entries}",
			    1024,0,1024,
			    "Baseline (adc counts)");



  //book histos
 
  nNoDatavsTime_ = bookProfile("nNoDatavsTime","nNoDatavsTime",
			       ";time;n_{noData}",
			       0,
			       42241,//total number of channels 
			       "Time",
			       "n_{noData}");

  nLowRangevsTime_ = bookProfile("nLowRangevsTime","nLowRangevsTime",
				 ";time;n_{LowRange}",
				 0,
				 42241,//total number of channels 
				 "Time",
				 "n_{LowRange}");

  nHighRangevsTime_ = bookProfile("nHighRangevsTime","nHighRangevsTime",
				  ";time;n_{HighRange}",
				  0,
				  42241,//total number of channels 
				  "Time",
				  "n_{HighRange}");


  nMinZerovsTime_ = bookProfile("nMinZerovsTime","nMinZerovsTime",
				";time;n_{MinZero}",
				0,
				42241,//total number of channels 
				"Time",
				"n_{MinZero}");

  nMaxSatvsTime_ = bookProfile("nMaxSatvsTime","nMaxSatvsTime",
			       ";time;n_{MaxSat}",
			       0,
			       42241,//total number of channels 
			       "Time",
			       "n_{MaxSat}");
  
  nLowPbvsTime_ = bookProfile("nLowPbvsTime","nLowPbvsTime",
				   ";time;n_{lowPb}",
				   0,
				   42241,//total number of channels 
				   "Time",
				   "n_{lowPb}");

  nHighPbvsTime_ = bookProfile("nHighPbvsTime","nHighPbvsTime",
				";time;n_{highPb}",
				    0,
				    42241,//total number of channels 
				    "Time",
				    "n_{highPb}");

 
  nOutOfSyncvsTime_ = bookProfile("nOutOfSyncvsTime","nOutOfSyncvsTime",
				  ";time;n_{OOS}",
				  0,
				  42241,//total number of channels 
				  "Time",
				  "n_{OOS}");

  nOtherPbsvsTime_ = bookProfile("nOtherPbsvsTime","nOtherPbsvsTime",
				 ";time;n_{OtherPbs}",
				 0,
				 42241,//total number of channels 
				 "Time",
				 "n_{OtherPbs}");

  nApvErrorBitvsTime_ = bookProfile("nApvErrorBitvsTime","nApvErrorBitvsTime",
				    ";time;n_{APVerror}",
				    0,
				    42241,//total number of channels 
				    "Time",
				    "n_{APVerror}");

  nApvAddressErrorvsTime_ = bookProfile("nApvAddressErrorvsTime","nApvAddressErrorvsTime",
					";time;n_{APVAddressError}",
					0,
					42241,//total number of channels 
					"Time",
					"n_{APVAddressError}");

  nNegativePedsvsTime_ = bookProfile("nNegativePedsvsTime","nNegativePedsvsTime",
				     ";time;n_{negPeds}",
				     0,
				     42241,//total number of channels 
				     "Time",
				     "n_{negPeds}");

  

  //book histos


  meanDigitalLowvsTime_ = bookProfile("meanDigitalLowvsTime","meanDigitalLowvsTime",
				     ";time;meanDigitalLow",
				     0,
				     800,//maximum for digitalLow 
				     "Time",
				     "meanDigitalLow");






  LogInfo("SPYHistograms") << " ---- Toplevel histograms book ! Number of MEs : " 
			   << dqm_->getMEs().size()
			   << std::endl;
  

}

void SPYHistograms::bookFEDHistograms(const unsigned int fedId,
				      const Errors & aErr,
				      bool doAll)
{
  if (!histosBooked_[fedId]) {
    //will do that only once

    //only if at least one error corresponds to enabled histo....
    if (!doAll && !(
		   (aErr.hasNoData && histogramConfig_["NoDataDetailed"].enabled) ||
		   (aErr.hasLowRange && histogramConfig_["LowRangeDetailed"].enabled) ||
		   (aErr.hasHighRange && histogramConfig_["HighRangeDetailed"].enabled) ||
		   (aErr.hasMinZero && histogramConfig_["MinZeroDetailed"].enabled) ||
		   (aErr.hasMaxSat && histogramConfig_["MaxSatDetailed"].enabled) ||
		   (aErr.hasLowPb && histogramConfig_["LowPbDetailed"].enabled) ||
		   (aErr.hasHighPb && histogramConfig_["HighPbDetailed"].enabled) ||
		   (aErr.hasOOS && histogramConfig_["OutOfSyncDetailed"].enabled) ||
		   (aErr.hasOtherPbs && histogramConfig_["OtherPbsDetailed"].enabled) ||
		   ((aErr.hasErrorBit0 ||aErr.hasErrorBit1)  && histogramConfig_["ApvErrorBitDetailed"].enabled) ||
		   ((aErr.hasAPVAddressError0 || aErr.hasAPVAddressError1) && histogramConfig_["ApvAddressErrorDetailed"].enabled) ||
		   (aErr.hasNegPeds && histogramConfig_["NegativePedsDetailed"].enabled)
		   )
	) return;
    


    SiStripFedKey fedKey(fedId,0,0,0);
    std::stringstream fedIdStream;
    fedIdStream << fedId;
    dqm_->setCurrentFolder(fedKey.path());
    

    noDataDetailed_[fedId] = bookHistogram("NoDataDetailed","NoDataForFED"+fedIdStream.str(),
					   "No data for FED ID "+fedIdStream.str(),
					   sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
					   "Channel Index"
					   );

    lowRangeDetailed_[fedId] = bookHistogram("LowRangeDetailed","LowRangeForFED"+fedIdStream.str(),
					     "Max-min<VAL for FED ID "+fedIdStream.str(),
					     sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
					     "Channel Index"
					     );

    highRangeDetailed_[fedId] = bookHistogram("HighRangeDetailed","HighRangeForFED"+fedIdStream.str(),
					      "Max-min>VAL for FED ID "+fedIdStream.str(),
					      sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
					      "Channel Index"
					      );

    minZeroDetailed_[fedId] = bookHistogram("MinZeroDetailed","MinZeroForFED"+fedIdStream.str(),
					    "min=0 for FED ID "+fedIdStream.str(),
					    sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
					    "Channel Index"
					    );

    maxSatDetailed_[fedId] = bookHistogram("MaxSatDetailed","MaxSatForFED"+fedIdStream.str(),
					   "max=1023 for FED ID "+fedIdStream.str(),
					   sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
					   "Channel Index"
					   );

    lowPbDetailed_[fedId] = bookHistogram("LowPbDetailed","LowPbForFED"+fedIdStream.str(),
					  "Pb with digitalLow for FED ID "+fedIdStream.str(),
					  sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
					  "Channel Index"
					  );

    highPbDetailed_[fedId] = bookHistogram("HighPbDetailed","HighPbForFED"+fedIdStream.str(),
					   "Pb with digitalHigh for FED ID "+fedIdStream.str(),
					   sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
					   "Channel Index"
					   );

    outOfSyncDetailed_[fedId] = bookHistogram("OutOfSyncDetailed","OutOfSyncForFED"+fedIdStream.str(),
					      "Out-of-sync for FED ID "+fedIdStream.str(),
					      sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
					      "Channel Index"
					      );

    otherPbsDetailed_[fedId] = bookHistogram("OtherPbsDetailed","OtherPbsForFED"+fedIdStream.str(),
					     "Other issues for FED ID "+fedIdStream.str(),
					     sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
					     "Channel Index"
					     );

    apvErrorBitDetailed_[fedId] = bookHistogram("ApvErrorBitDetailed","ApvErrorBitForFED"+fedIdStream.str(),
						"APV error for FED ID "+fedIdStream.str(),
						2*sistrip::FEDCH_PER_FED,0,2*sistrip::FEDCH_PER_FED,
						"APV Index"
						);

    apvAddressErrorDetailed_[fedId] = bookHistogram("ApvAddressErrorDetailed","ApvAddressErrorForFED"+fedIdStream.str(),
						    "APV address error for FED ID "+fedIdStream.str(),
						    2*sistrip::FEDCH_PER_FED,0,2*sistrip::FEDCH_PER_FED,
						    "APV Index"
						    );

    negativePedsDetailed_[fedId] = bookHistogram("NegativePedsDetailed","NegativePedsForFED"+fedIdStream.str(),
						 "Negative ped subtr for FED ID "+fedIdStream.str(),
						 sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
						 "Channel Index"
						 );



    positionOfFirstHeaderBitDetailed_[fedId] = bookHistogram("PositionOfFirstHeaderBitDetailed",
							     "PositionOfFirstHeaderBitForFED"+fedIdStream.str(),
							     "FED ID "+fedIdStream.str(),
							     300,0,300,
							     "first header bit"
							     );

    positionOfFirstTrailerBitDetailed_[fedId] = bookHistogram("PositionOfFirstTrailerBitDetailed",
							      "PositionOfFirstTrailerBitForFED"+fedIdStream.str(),
							      "FED ID "+fedIdStream.str(),
							      15,285,300,
							      "first trailer bit"
							      );
					    
    distanceHeaderTrailerDetailed_[fedId] = bookHistogram("DistanceHeaderTrailerDetailed",
							  "DistanceHeaderTrailerForFED"+fedIdStream.str(),
							  "FED ID "+fedIdStream.str(),
							  300,0,300,
							  "trailer - header"
							  );


    histosBooked_[fedId] = true;

  }


}

void SPYHistograms::bookAllFEDHistograms()
{
  //get FED IDs
  const unsigned int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const unsigned int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;
  //book them
  for (unsigned int iFed = siStripFedIdMin; iFed <= siStripFedIdMax; iFed++) {
    //dummy error object
    Errors lError;
    bookFEDHistograms(iFed,lError,true);
  }
}
