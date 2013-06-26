#include "EventFilter/CastorRawToDigi/interface/CastorCtdcUnpacker.h"
#include "EventFilter/CastorRawToDigi/interface/CastorCTDCHeader.h"
#include "EventFilter/CastorRawToDigi/interface/CastorCORData.h"
#include "DataFormats/HcalDetId/interface/HcalOtherDetId.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "DataFormats/HcalDigi/interface/CastorDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveSample.h"
#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
using namespace std;

CastorCtdcUnpacker::CastorCtdcUnpacker(int sourceIdOffset, int beg, int end) : sourceIdOffset_(sourceIdOffset)
{
 if ( beg >= 0 && beg <= CastorDataFrame::MAXSAMPLES -1 ) {
 	startSample_ = beg;
 } else {
 	startSample_ = 0;
 }
 if ( end >= 0 && end <= CastorDataFrame::MAXSAMPLES -1 && end >= beg ) {
 	endSample_ = end;
 } else {
 	endSample_ = CastorDataFrame::MAXSAMPLES -1;
 }
}


void CastorCtdcUnpacker::unpack(const FEDRawData& raw, const CastorElectronicsMap& emap,
			  CastorRawCollections& colls, HcalUnpackerReport& report) {

  if (raw.size()<16) {
    edm::LogWarning("Invalid Data") << "Empty/invalid DCC data, size = " << raw.size();
    return;
  }

  // get the CTDC header
  const CastorCTDCHeader* ctdcHeader=(const CastorCTDCHeader*)(raw.data());
  int ctdcid=ctdcHeader->getSourceId()-sourceIdOffset_;

  // space for unpacked data from one COR
  std::vector<unsigned short> precdata(CastorCORData::CHANNELS_PER_SPIGOT*CastorCORData::MAXIMUM_SAMPLES_PER_CHANNEL);
  std::vector<unsigned short> trigdata(CastorCORData::CHANNELS_PER_SPIGOT*CastorCORData::MAXIMUM_SAMPLES_PER_CHANNEL);
  std::vector<unsigned char>  preclen(CastorCORData::CHANNELS_PER_SPIGOT);
  std::vector<unsigned char>  triglen(CastorCORData::CHANNELS_PER_SPIGOT);
  
  // walk through the COR data...
  CastorCORData cor;
  for (int spigot=0; spigot<CastorCTDCHeader::SPIGOT_COUNT; spigot++) {
    if (!ctdcHeader->getSpigotPresent(spigot)) continue;

    int retval=ctdcHeader->getSpigotData(spigot,cor,raw.size());
    if (retval!=0) {
      if (retval==-1) {
	edm::LogWarning("Invalid Data") << "Invalid COR data (data beyond payload size) observed on spigot " << spigot << " of CTDC with source id " << ctdcHeader->getSourceId();
	report.countSpigotFormatError();
      }
      continue;
    }
    // check
    if (!cor.check()) {
      edm::LogWarning("Invalid Data") << "Invalid COR data observed on spigot " << spigot << " of CTDC with source id " << ctdcHeader->getSourceId();
      report.countSpigotFormatError();
      continue;
    }
    if (cor.isHistogramEvent()) {
      edm::LogWarning("Invalid Data") << "Histogram data passed to non-histogram unpacker on spigot " << spigot << " of CTDC with source id " << ctdcHeader->getSourceId();
      continue;

    }
    // calculate "real" number of presamples
    int nps=cor.getNPS()-startSample_;

    // new: do not use get pointers .. instead make use of CastorCORData::unpack   

     cor.unpack(&(preclen[0]),&(precdata[0]),
			    &(triglen[0]),&(trigdata[0]));
 
    
    /// work through all channels 
   int ichan;
   for ( ichan = 0; ichan<CastorCORData::CHANNELS_PER_SPIGOT; ichan++) {
	   if ( preclen[ichan] == 0 || preclen[ichan] & 0xc0 ) continue;
		int fiber = ichan/3;
		int fiberchan = ichan%3;
        // lookup the right channel
	    CastorElectronicsId partialEid(fiberchan,fiber+1,spigot,ctdcid);
	    // does this partial id exist?
	    CastorElectronicsId eid;
	    HcalGenericDetId did;
	    bool found;
 	    found = emap.lookup(partialEid,eid,did);

		if (found) {			
			CastorDataFrame digi = CastorDataFrame(HcalCastorDetId(did));
			// set parameters - presamples
			digi.setPresamples(nps);

			int ntaken = 0;
			for ( int sample = startSample_; sample <= endSample_; sample++ ) {
					digi.setSample(ntaken,precdata[ichan*CastorCORData::MAXIMUM_SAMPLES_PER_CHANNEL+sample]);
					ntaken++;
			}
			digi.setSize(ntaken);
			colls.castorCont->push_back(digi);			
		} else {
			report.countUnmappedDigi();
			if (unknownIds_.find(partialEid)==unknownIds_.end()) {
			  edm::LogWarning("CASTOR") << "CastorCtdcUnpacker: No match found for electronics partialEid :" << partialEid;
			  unknownIds_.insert(partialEid);
		    }
		}
    }

  }
}


