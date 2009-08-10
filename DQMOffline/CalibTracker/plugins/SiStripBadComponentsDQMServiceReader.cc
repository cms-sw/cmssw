#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CondFormats/DataRecord/interface/SiStripBadStripRcd.h"

#include "DQMOffline/CalibTracker/plugins/SiStripBadComponentsDQMServiceReader.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>

#include <boost/lexical_cast.hpp>

SiStripBadComponentsDQMServiceReader::SiStripBadComponentsDQMServiceReader( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<bool>("printDebug",true)){}

SiStripBadComponentsDQMServiceReader::~SiStripBadComponentsDQMServiceReader(){}

void SiStripBadComponentsDQMServiceReader::analyze( const edm::Event& e, const edm::EventSetup& iSetup)
{
  uint32_t FedErrorMask = 1;     // bit 0
  uint32_t DigiErrorMask = 2;    // bit 1
  uint32_t ClusterErrorMask = 4; // bit 2

  edm::ESHandle<SiStripBadStrip> SiStripBadStrip_;
  iSetup.get<SiStripBadStripRcd>().get(SiStripBadStrip_);
  edm::LogInfo("SiStripBadComponentsDQMServiceReader") << "[SiStripBadComponentsDQMServiceReader::analyze] End Reading SiStripBadStrip" << std::endl;

  std::vector<uint32_t> detid;
  SiStripBadStrip_->getDetIds(detid);

  std::stringstream ss;

  if (printdebug_) {

    // ss << " detid" << " \t\t\t" << "FED error" << " \t" << "Digi test failed" << " \t" << "Cluster test failed" << std::endl;

    ss << "  detid" << " \t\t" << "Errors" << std::endl;

    for (size_t id=0;id<detid.size();id++) {
      SiStripBadStrip::Range range=SiStripBadStrip_->getRange(detid[id]);

      for(int it=0;it<range.second-range.first;it++){
	unsigned int value=(*(range.first+it));
	// 	  edm::LogInfo("SiStripBadComponentsDQMServiceReader")  << "detid " << detid[id] << " \t"
	// 						 << " firstBadStrip " <<  SiStripBadStrip_->decode(value).firstStrip << "\t "
	// 						 << " NconsecutiveBadStrips " << SiStripBadStrip_->decode(value).range << "\t "
	// 						 << " flag " << SiStripBadStrip_->decode(value).flag << "\t "
	// 						 << " packed integer " <<  std::hex << value << std::dec << "\t "

	ss << detid[id] << " \t";
	uint32_t flag = boost::lexical_cast<uint32_t>(SiStripBadStrip_->decode(value).flag);

// 	std::cout << "fed error = " << ((flag & FedErrorMask) == FedErrorMask) << ", for flag = " << flag << std::endl;
// 	std::cout << "digi error = " << ((flag & DigiErrorMask) == DigiErrorMask) << ", for flag = " << flag << ", with (flag & DigiErrorMask) = " << (flag & DigiErrorMask) << std::endl;
// 	std::cout << "cluster error = " << ((flag & ClusterErrorMask) == ClusterErrorMask) << ", for flag = " << flag << std::endl;

	printError( ss, ((flag & FedErrorMask) == FedErrorMask), "Fed error, " );
	printError( ss, ((flag & DigiErrorMask) == DigiErrorMask), "Digi error, " );
	printError( ss, ((flag & ClusterErrorMask) == ClusterErrorMask), "Cluster error" );
	ss << std::endl;
      }
    }
  }
  edm::LogInfo("SiStripBadComponentsDQMServiceReader") << ss.str();
}

void SiStripBadComponentsDQMServiceReader::printError( std::stringstream & ss, const bool error, const std::string & errorText)
{
  if( error ) {
    ss << errorText << "\t ";
  }
  else {
    ss << "\t\t ";
  }
}
