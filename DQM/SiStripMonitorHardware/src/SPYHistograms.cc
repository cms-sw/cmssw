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
}

SPYHistograms::~SPYHistograms()
{
}
 

void 
SPYHistograms::initialise(const edm::ParameterSet& iConfig,
			  std::ostringstream* pDebugStream){

  getConfigForHistogram(nNoData_,"nNoData",iConfig,pDebugStream);
  getConfigForHistogram(nLowRange_,"nLowRange",iConfig,pDebugStream);
  getConfigForHistogram(nHighRange_,"nHighRange",iConfig,pDebugStream);
  getConfigForHistogram(nMinZero_,"nMinZero",iConfig,pDebugStream);
  getConfigForHistogram(nMaxSat_,"nMaxSat",iConfig,pDebugStream);
  getConfigForHistogram(nLowPb_,"nLowPb",iConfig,pDebugStream);
  getConfigForHistogram(nHighPb_,"nHighPb",iConfig,pDebugStream);
  getConfigForHistogram(nOutOfSync_,"nOutOfSync",iConfig,pDebugStream);
  getConfigForHistogram(nOtherPbs_,"nOtherPbs",iConfig,pDebugStream);
  getConfigForHistogram(nApvErrorBit_,"nApvErrorBit",iConfig,pDebugStream);
  getConfigForHistogram(nApvAddressError_,"nApvAddressError",iConfig,pDebugStream);
  getConfigForHistogram(nNegativePeds_,"nNegativePeds",iConfig,pDebugStream);

  getConfigForHistogram(nNoDatavsTime_,"nNoDatavsTime",iConfig,pDebugStream);
  getConfigForHistogram(nLowRangevsTime_,"nLowRangevsTime",iConfig,pDebugStream);
  getConfigForHistogram(nHighRangevsTime_,"nHighRangevsTime",iConfig,pDebugStream);
  getConfigForHistogram(nMinZerovsTime_,"nMinZerovsTime",iConfig,pDebugStream);
  getConfigForHistogram(nMaxSatvsTime_,"nMaxSatvsTime",iConfig,pDebugStream);
  getConfigForHistogram(nLowPbvsTime_,"nLowPbvsTime",iConfig,pDebugStream);
  getConfigForHistogram(nHighPbvsTime_,"nHighPbvsTime",iConfig,pDebugStream);
  getConfigForHistogram(nOutOfSyncvsTime_,"nOutOfSyncvsTime",iConfig,pDebugStream);
  getConfigForHistogram(nOtherPbsvsTime_,"nOtherPbsvsTime",iConfig,pDebugStream);
  getConfigForHistogram(nApvErrorBitvsTime_,"nApvErrorBitvsTime",iConfig,pDebugStream);
  getConfigForHistogram(nApvAddressErrorvsTime_,"nApvAddressErrorvsTime",iConfig,pDebugStream);
  getConfigForHistogram(nNegativePedsvsTime_,"nNegativePedsvsTime",iConfig,pDebugStream);

  getConfigForHistogram(meanDigitalLowvsTime_,"meanDigitalLowvsTime",iConfig,pDebugStream);


  getConfigForHistogram(noData_,"NoData",iConfig,pDebugStream);
  getConfigForHistogram(lowRange_,"LowRange",iConfig,pDebugStream);
  getConfigForHistogram(highRange_,"HighRange",iConfig,pDebugStream);
  getConfigForHistogram(minZero_,"MinZero",iConfig,pDebugStream);
  getConfigForHistogram(maxSat_,"MaxSat",iConfig,pDebugStream);
  getConfigForHistogram(lowPb_,"LowPb",iConfig,pDebugStream);
  getConfigForHistogram(highPb_,"HighPb",iConfig,pDebugStream);
  getConfigForHistogram(outOfSync_,"OutOfSync",iConfig,pDebugStream);
  getConfigForHistogram(otherPbs_,"OtherPbs",iConfig,pDebugStream);
  getConfigForHistogram(apvErrorBit_,"ApvErrorBit",iConfig,pDebugStream);
  getConfigForHistogram(apvAddressError_,"ApvAddressError",iConfig,pDebugStream);
  getConfigForHistogram(negativePeds_,"NegativePeds",iConfig,pDebugStream);

  getConfigForHistogram(frameRange_,"FrameRange",iConfig,pDebugStream);
  getConfigForHistogram(frameMin_,"FrameMin",iConfig,pDebugStream);
  getConfigForHistogram(frameMax_,"FrameMax",iConfig,pDebugStream);
  getConfigForHistogram(baseline_,"Baseline",iConfig,pDebugStream);

  getConfigForHistogram(noDataDetailed_,"NoDataDetailed",iConfig,pDebugStream);
  getConfigForHistogram(lowRangeDetailed_,"LowRangeDetailed",iConfig,pDebugStream);
  getConfigForHistogram(highRangeDetailed_,"HighRangeDetailed",iConfig,pDebugStream);
  getConfigForHistogram(minZeroDetailed_,"MinZeroDetailed",iConfig,pDebugStream);
  getConfigForHistogram(maxSatDetailed_,"MaxSatDetailed",iConfig,pDebugStream);
  getConfigForHistogram(lowPbDetailed_,"LowPbDetailed",iConfig,pDebugStream);
  getConfigForHistogram(highPbDetailed_,"HighPbDetailed",iConfig,pDebugStream);
  getConfigForHistogram(outOfSyncDetailed_,"OutOfSyncDetailed",iConfig,pDebugStream);
  getConfigForHistogram(otherPbsDetailed_,"OtherPbsDetailed",iConfig,pDebugStream);
  getConfigForHistogram(apvErrorBitDetailed_,"ApvErrorBitDetailed",iConfig,pDebugStream);
  getConfigForHistogram(apvAddressErrorDetailed_,"ApvAddressErrorDetailed",iConfig,pDebugStream);
  getConfigForHistogram(negativePedsDetailed_,"NegativePedsDetailed",iConfig,pDebugStream);

  getConfigForHistogram(positionOfFirstHeaderBitDetailed_,"PositionOfFirstHeaderBitDetailed",iConfig,pDebugStream);
  getConfigForHistogram(positionOfFirstTrailerBitDetailed_,"PositionOfFirstTrailerBitDetailed",iConfig,pDebugStream);
  getConfigForHistogram(distanceHeaderTrailerDetailed_,"DistanceHeaderTrailerDetailed",iConfig,pDebugStream);

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
  //removed: tomas
  //bookFEDHistograms(aFedId,aErr);

  if (!aErr.hasNoData) {
    fillHistogram(frameRange_,sistrip::SpyUtilities::range(aFrame));
    fillHistogram(frameMin_,aFrame.digitalLow);
    fillHistogram(frameMax_,aFrame.digitalHigh);
    fillHistogram(baseline_,aFrame.baseline);
  }

  if (aErr.hasNoData) {
    fillHistogram(noDataDetailedMap_[aFedId],aFedChannel);
    //fillHistogram(noData_,24*aFedId+static_cast<unsigned int>(aFedChannel/4.));
    fillHistogram(noData_,static_cast<unsigned int>(aFedChannel/4.));
  }

  if (aErr.hasLowRange) { 
    fillHistogram(lowRangeDetailedMap_[aFedId],aFedChannel);
  }
  if (aErr.hasHighRange) { 
    fillHistogram(highRangeDetailedMap_[aFedId],aFedChannel);
  }
  if (aErr.hasMinZero) {
    fillHistogram(minZeroDetailedMap_[aFedId],aFedChannel);
  }
  if (aErr.hasMaxSat) {
    fillHistogram(maxSatDetailedMap_[aFedId],aFedChannel);
  }
  if (aErr.hasLowPb) {
    fillHistogram(lowPbDetailedMap_[aFedId],aFedChannel);
  }
  if (aErr.hasHighPb) {
    fillHistogram(highPbDetailedMap_[aFedId],aFedChannel);
  }
  if (aErr.hasOOS) {
    fillHistogram(outOfSyncDetailedMap_[aFedId],aFedChannel);
  }
  if (aErr.hasOtherPbs) {
    fillHistogram(otherPbsDetailedMap_[aFedId],aFedChannel);
  }
  if (aErr.hasErrorBit0) {
    fillHistogram(apvErrorBitDetailedMap_[aFedId],2*aFedChannel);
  }
  if (aErr.hasErrorBit1) {
    fillHistogram(apvErrorBitDetailedMap_[aFedId],2*aFedChannel+1);
  }
  if (aErr.hasAPVAddressError0) {
    fillHistogram(apvAddressErrorDetailedMap_[aFedId],2*aFedChannel);
  }
  if (aErr.hasAPVAddressError1) {
    fillHistogram(apvAddressErrorDetailedMap_[aFedId],2*aFedChannel+1);
  }
  if (aErr.hasNegPeds) {
    fillHistogram(negativePedsDetailedMap_[aFedId],aFedChannel);
  }

  fillHistogram(positionOfFirstHeaderBitDetailedMap_[aFedId],aFrame.firstHeaderBit);
  fillHistogram(positionOfFirstTrailerBitDetailedMap_[aFedId],aFrame.firstTrailerBit);
  if (aFrame.firstTrailerBit != sistrip::SPY_SAMPLES_PER_CHANNEL &&
      aFrame.firstHeaderBit != sistrip::SPY_SAMPLES_PER_CHANNEL)
    fillHistogram(distanceHeaderTrailerDetailedMap_[aFedId],aFrame.firstTrailerBit-aFrame.firstHeaderBit);
  else fillHistogram(distanceHeaderTrailerDetailedMap_[aFedId],0);
}


void SPYHistograms::bookTopLevelHistograms(DQMStore::IBooker & ibooker)
{
  //get FED IDs
  const unsigned int siStripFedIdMin = sistrip::FED_ID_MIN;
  const unsigned int siStripFedIdMax = sistrip::FED_ID_MAX;

  //book FED level histograms: resize boolean so 
  //they will be booked only once for the first event.
  histosBooked_.resize(siStripFedIdMax+1,false);


  //book histos
  bookHistogram(ibooker , nNoData_,"nNoData",
		";n_{noData};n_{entries}",
		"n_{noData}");

  bookHistogram(ibooker , nLowRange_,"nLowRange",
		";n_{LowRange};n_{entries}",
		"n_{LowRange}");

  bookHistogram(ibooker , nHighRange_,"nHighRange",
		";n_{HighRange};n_{entries}",
		"n_{HighRange}");

  bookHistogram(ibooker , nMinZero_,"nMinZero",
		";n_{MinZero};n_{entries}",
		"n_{MinZero}");

  bookHistogram(ibooker , nMaxSat_,"nMaxSat",
		";n_{MaxSat};n_{entries}",
		"n_{MaxSat}");
  
  bookHistogram(ibooker , nLowPb_,"nLowPb",
		";n_{lowPb};n_{entries}",
		"n_{lowPb}");

  bookHistogram(ibooker , nHighPb_,"nHighPb",
		";n_{highPb};n_{entries}",
		"n_{highPb}");
  
  bookHistogram(ibooker , nOutOfSync_,"nOutOfSync",
		";n_{OOS};n_{entries}",
		"n_{OOS}");

  bookHistogram(ibooker , nOtherPbs_,"nOtherPbs",
		";n_{OtherPbs};n_{entries}",
		"n_{OtherPbs}");

  bookHistogram(ibooker , nApvErrorBit_,"nApvErrorBit",
		";n_{APVerror};n_{entries}",
		"n_{APVerror}"
		);

  bookHistogram(ibooker , nApvAddressError_,"nApvAddressError",
		";n_{APVAddressError};n_{entries}",
		"n_{APVAddressError}"
		);

  bookHistogram(ibooker , nNegativePeds_,"nNegativePeds",
		";n_{negPeds};n_{entries}",
		"n_{negPeds}"
		);
  


  bookHistogram(ibooker , noData_,"NoData",
		";delayFPGA;n_{noData}",
		//24*(siStripFedIdMax+1)-24*siStripFedIdMin,
		//24*siStripFedIdMin,24*(siStripFedIdMax+1),
		24,0,24,
		"delayFPGA index"
		);

  bookHistogram(ibooker , lowRange_,"LowRange",
		";fedId;n_{LowRange}",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,
		"FED-ID"
		);
  
  bookHistogram(ibooker , highRange_,"HighRange",
		";fedId;n_{HighRange}",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,
		"FED-ID"
		);

  bookHistogram(ibooker , minZero_,"MinZero",
		";fedId;n_{MinZero}",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,
		"FED-ID"
		);

  bookHistogram(ibooker , maxSat_,"MaxSat",
		";fedId;n_{MaxSat}",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,
		"FED-ID"
		);
  
  bookHistogram(ibooker , lowPb_,"LowPb",
		";fedId;n_{lowPb}",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,
		"FED-ID"
		);

  bookHistogram(ibooker , highPb_,"HighPb",
		";fedId;n_{highPb}",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,
		"FED-ID"
		);
  
  bookHistogram(ibooker , outOfSync_,"OutOfSync",
		";fedId;n_{OOS}",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,
		"FED-ID"
		);

  bookHistogram(ibooker , otherPbs_,"OtherPbs",
		";fedId;n_{OtherPbs}",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,
		"FED-ID"
		);

  bookHistogram(ibooker , apvErrorBit_,"ApvErrorBit",
		";fedId;n_{APVerror}",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,
		"FED-ID"
		);

  bookHistogram(ibooker , apvAddressError_,"ApvAddressError",
		";fedId;n_{APVAddressError}",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,
		"FED-ID"
		);

  bookHistogram(ibooker , negativePeds_,"NegativePeds",
		";fedId;n_{negPeds}",
		siStripFedIdMax-siStripFedIdMin+1,
		siStripFedIdMin-0.5,siStripFedIdMax+0.5,
		"FED-ID"
		);
  
  bookHistogram(ibooker , frameRange_,"FrameRange",
		";range;n_{entries}",
		1024,0,1024,
		"range (adc counts)");

  bookHistogram(ibooker , frameMin_,"FrameMin",
		";Zero-light level;n_{entries}",
		800,0,800,
		"Zero-light level (adc counts)");

  bookHistogram(ibooker , frameMax_,"FrameMax",
		";Tick height;n_{entries}",
		824,200,1024,
		"Tick height (adc counts)");

  bookHistogram(ibooker , baseline_,"Baseline",
		";Baseline;n_{entries}",
		1024,0,1024,
		"Baseline (adc counts)");



  //book histos
 
  bookProfile(ibooker , nNoDatavsTime_,"nNoDatavsTime",
	      ";time;n_{noData}",
	      0,
	      42241,//total number of channels 
	      "Time",
	      "n_{noData}");

  bookProfile(ibooker , nLowRangevsTime_,"nLowRangevsTime",
	      ";time;n_{LowRange}",
	      0,
	      42241,//total number of channels 
	      "Time",
	      "n_{LowRange}");

  bookProfile(ibooker , nHighRangevsTime_,"nHighRangevsTime",
	      ";time;n_{HighRange}",
	      0,
	      42241,//total number of channels 
	      "Time",
	      "n_{HighRange}");


  bookProfile(ibooker , nMinZerovsTime_,"nMinZerovsTime",
	      ";time;n_{MinZero}",
	      0,
	      42241,//total number of channels 
	      "Time",
	      "n_{MinZero}");

  bookProfile(ibooker , nMaxSatvsTime_,"nMaxSatvsTime",
	      ";time;n_{MaxSat}",
	      0,
	      42241,//total number of channels 
	      "Time",
	      "n_{MaxSat}");
  
  bookProfile(ibooker , nLowPbvsTime_,"nLowPbvsTime",
	      ";time;n_{lowPb}",
	      0,
	      42241,//total number of channels 
	      "Time",
	      "n_{lowPb}");

  bookProfile(ibooker , nHighPbvsTime_,"nHighPbvsTime",
	      ";time;n_{highPb}",
	      0,
	      42241,//total number of channels 
	      "Time",
	      "n_{highPb}");

 
  bookProfile(ibooker , nOutOfSyncvsTime_,"nOutOfSyncvsTime",
	      ";time;n_{OOS}",
	      0,
	      42241,//total number of channels 
	      "Time",
	      "n_{OOS}");

  bookProfile(ibooker , nOtherPbsvsTime_,"nOtherPbsvsTime",
	      ";time;n_{OtherPbs}",
	      0,
	      42241,//total number of channels 
	      "Time",
	      "n_{OtherPbs}");

  bookProfile(ibooker , nApvErrorBitvsTime_,"nApvErrorBitvsTime",
	      ";time;n_{APVerror}",
	      0,
	      42241,//total number of channels 
	      "Time",
	      "n_{APVerror}");

  bookProfile(ibooker , nApvAddressErrorvsTime_,"nApvAddressErrorvsTime",
	      ";time;n_{APVAddressError}",
	      0,
	      42241,//total number of channels 
	      "Time",
	      "n_{APVAddressError}");

  bookProfile(ibooker , nNegativePedsvsTime_,"nNegativePedsvsTime",
	      ";time;n_{negPeds}",
	      0,
	      42241,//total number of channels 
	      "Time",
	      "n_{negPeds}");

  

  //book histos


  bookProfile(ibooker , meanDigitalLowvsTime_,"meanDigitalLowvsTime",
	      ";time;meanDigitalLow",
	      0,
	      800,//maximum for digitalLow 
	      "Time",
	      "meanDigitalLow");
}

void SPYHistograms::bookFEDHistograms(DQMStore::IBooker & ibooker , const unsigned int fedId,
				      const Errors & aErr,
				      bool doAll)
{
  if (!histosBooked_[fedId]) {
    //will do that only once

    //only if at least one error corresponds to enabled histo....
    if (!doAll && !(
		   (aErr.hasNoData && noDataDetailed_.enabled) ||
		   (aErr.hasLowRange && lowRangeDetailed_.enabled) ||
		   (aErr.hasHighRange && highRangeDetailed_.enabled) ||
		   (aErr.hasMinZero && minZeroDetailed_.enabled) ||
		   (aErr.hasMaxSat && maxSatDetailed_.enabled) ||
		   (aErr.hasLowPb && lowPbDetailed_.enabled) ||
		   (aErr.hasHighPb && highPbDetailed_.enabled) ||
		   (aErr.hasOOS && outOfSyncDetailed_.enabled) ||
		   (aErr.hasOtherPbs && otherPbsDetailed_.enabled) ||
		   ((aErr.hasErrorBit0 ||aErr.hasErrorBit1)  && apvErrorBitDetailed_.enabled) ||
		   ((aErr.hasAPVAddressError0 || aErr.hasAPVAddressError1) && apvAddressErrorDetailed_.enabled) ||
		   (aErr.hasNegPeds && negativePedsDetailed_.enabled)
		   )
	) return;
    


    SiStripFedKey fedKey(fedId,0,0,0);
    std::stringstream fedIdStream;
    fedIdStream << fedId;
    ibooker.setCurrentFolder(fedKey.path());
    

    bookHistogram(ibooker , noDataDetailed_,
		  noDataDetailedMap_[fedId],
		  "NoDataForFED"+fedIdStream.str(),
		  "No data for FED ID "+fedIdStream.str(),
		  sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
		  "Channel Index"
		  );

    bookHistogram(ibooker , lowRangeDetailed_,
		  lowRangeDetailedMap_[fedId],
		  "LowRangeForFED"+fedIdStream.str(),
		  "Max-min<VAL for FED ID "+fedIdStream.str(),
		  sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
		  "Channel Index"
		  );

    bookHistogram(ibooker , highRangeDetailed_,
		  highRangeDetailedMap_[fedId],
		  "HighRangeForFED"+fedIdStream.str(),
		  "Max-min>VAL for FED ID "+fedIdStream.str(),
		  sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
		  "Channel Index"
		  );

    bookHistogram(ibooker , minZeroDetailed_,
		  minZeroDetailedMap_[fedId],
		  "MinZeroForFED"+fedIdStream.str(),
		  "min=0 for FED ID "+fedIdStream.str(),
		  sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
		  "Channel Index"
		  );

    bookHistogram(ibooker , maxSatDetailed_,
		  maxSatDetailedMap_[fedId],
		  "MaxSatForFED"+fedIdStream.str(),
		  "max=1023 for FED ID "+fedIdStream.str(),
		  sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
		  "Channel Index"
		  );

    bookHistogram(ibooker , lowPbDetailed_,
		  lowPbDetailedMap_[fedId],
		  "LowPbForFED"+fedIdStream.str(),
		  "Pb with digitalLow for FED ID "+fedIdStream.str(),
		  sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
		  "Channel Index"
		  );

    bookHistogram(ibooker , highPbDetailed_,
		  highPbDetailedMap_[fedId],
		  "HighPbForFED"+fedIdStream.str(),
		  "Pb with digitalHigh for FED ID "+fedIdStream.str(),
		  sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
		  "Channel Index"
		  );

    bookHistogram(ibooker , outOfSyncDetailed_,
		  outOfSyncDetailedMap_[fedId],
		  "OutOfSyncForFED"+fedIdStream.str(),
		  "Out-of-sync for FED ID "+fedIdStream.str(),
		  sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
		  "Channel Index"
		  );

    bookHistogram(ibooker , otherPbsDetailed_,
		  otherPbsDetailedMap_[fedId],
		  "OtherPbsForFED"+fedIdStream.str(),
		  "Other issues for FED ID "+fedIdStream.str(),
		  sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
		  "Channel Index"
		  );

    bookHistogram(ibooker , apvErrorBitDetailed_,
		  apvErrorBitDetailedMap_[fedId],
		  "ApvErrorBitForFED"+fedIdStream.str(),
		  "APV error for FED ID "+fedIdStream.str(),
		  2*sistrip::FEDCH_PER_FED,0,2*sistrip::FEDCH_PER_FED,
		  "APV Index"
		  );

    bookHistogram(ibooker , apvAddressErrorDetailed_,
		  apvAddressErrorDetailedMap_[fedId],
		  "ApvAddressErrorForFED"+fedIdStream.str(),
		  "APV address error for FED ID "+fedIdStream.str(),
		  2*sistrip::FEDCH_PER_FED,0,2*sistrip::FEDCH_PER_FED,
		  "APV Index"
		  );

    bookHistogram(ibooker , negativePedsDetailed_,
		  negativePedsDetailedMap_[fedId],
		  "NegativePedsForFED"+fedIdStream.str(),
		  "Negative ped subtr for FED ID "+fedIdStream.str(),
		  sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
		  "Channel Index"
		  );



    bookHistogram(ibooker , positionOfFirstHeaderBitDetailed_,
		  positionOfFirstHeaderBitDetailedMap_[fedId],
		  "PositionOfFirstHeaderBitForFED"+fedIdStream.str(),
		  "FED ID "+fedIdStream.str(),
		  300,0,300,
		  "first header bit"
		  );

    bookHistogram(ibooker , positionOfFirstTrailerBitDetailed_,
		  positionOfFirstTrailerBitDetailedMap_[fedId],
		  "PositionOfFirstTrailerBitForFED"+fedIdStream.str(),
		  "FED ID "+fedIdStream.str(),
		  15,285,300,
		  "first trailer bit"
		  );
					    
    bookHistogram(ibooker , distanceHeaderTrailerDetailed_,
		  distanceHeaderTrailerDetailedMap_[fedId],
		  "DistanceHeaderTrailerForFED"+fedIdStream.str(),
		  "FED ID "+fedIdStream.str(),
		  300,0,300,
		  "trailer - header"
		  );


    histosBooked_[fedId] = true;

  }


}

void SPYHistograms::bookAllFEDHistograms(DQMStore::IBooker & ibooker)
{
  //get FED IDs
  const unsigned int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const unsigned int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;
  //book them
  for (unsigned int iFed = siStripFedIdMin; iFed <= siStripFedIdMax; iFed++) {
    //dummy error object
    Errors lError;
    bookFEDHistograms(ibooker , iFed,lError,true);
  }
}
