#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CondFormats/DataRecord/interface/SiStripBadStripRcd.h"

#include "DQMOffline/CalibTracker/plugins/SiStripBadComponentsDQMServiceReader.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>

#include <boost/lexical_cast.hpp>

using namespace std;

SiStripBadComponentsDQMServiceReader::SiStripBadComponentsDQMServiceReader( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<bool>("printDebug",true)){}

SiStripBadComponentsDQMServiceReader::~SiStripBadComponentsDQMServiceReader(){}

void SiStripBadComponentsDQMServiceReader::analyze( const edm::Event& e, const edm::EventSetup& iSetup)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  uint32_t FedErrorMask = 1;     // bit 0
  uint32_t DigiErrorMask = 2;    // bit 1
  uint32_t ClusterErrorMask = 4; // bit 2

  edm::ESHandle<SiStripBadStrip> SiStripBadStrip_;
  iSetup.get<SiStripBadStripRcd>().get(SiStripBadStrip_);
  edm::LogInfo("SiStripBadComponentsDQMServiceReader") << "[SiStripBadComponentsDQMServiceReader::analyze] End Reading SiStripBadStrip" << std::endl;

  std::vector<uint32_t> detid;
  SiStripBadStrip_->getDetIds(detid);

  std::stringstream ss;

  // ss << " detid" << " \t\t\t" << "FED error" << " \t" << "Digi test failed" << " \t" << "Cluster test failed" << std::endl;

  ss << "subdet  layer   stereo  side \t detId \t\t Errors" << std::endl;

  for (size_t id=0;id<detid.size();id++) {
    SiStripBadStrip::Range range=SiStripBadStrip_->getRange(detid[id]);

    for(int it=0;it<range.second-range.first;it++){
      unsigned int value=(*(range.first+it));
      ss << detIdToString(detid[id], tTopo) << "\t" << detid[id] << "\t";

      uint32_t flag = boost::lexical_cast<uint32_t>(SiStripBadStrip_->decode(value).flag);

      printError( ss, ((flag & FedErrorMask) == FedErrorMask), "Fed error, " );
      printError( ss, ((flag & DigiErrorMask) == DigiErrorMask), "Digi error, " );
      printError( ss, ((flag & ClusterErrorMask) == ClusterErrorMask), "Cluster error" );
      ss << std::endl;

      if (printdebug_) {
        ss << " firstBadStrip " <<  SiStripBadStrip_->decode(value).firstStrip << "\t "
           << " NconsecutiveBadStrips " << SiStripBadStrip_->decode(value).range << "\t " // << std::endl;
           << " flag " << SiStripBadStrip_->decode(value).flag << "\t "
           << " packed integer " <<  std::hex << value << std::dec << "\t " << std::endl;
      }
    }
    ss << std::endl;
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

string SiStripBadComponentsDQMServiceReader::detIdToString(const DetId & detid, const TrackerTopology* tTopo)
{
  std::string detector;
  int layer = 0;
  int stereo = 0;
  int side = -1;

  // Using the operator[] if the element does not exist it is created with the default value. That is 0 for integral types.
  switch (detid.subdetId()) {
  case StripSubdetector::TIB:
    {
      
      detector = "TIB";
      layer = tTopo->tibLayer(detid.rawId());
      stereo = tTopo->tibStereo(detid.rawId());
      break;
    }
  case StripSubdetector::TOB:
    {
      
      detector = "TOB";
      layer = tTopo->tobLayer(detid.rawId());
      stereo = tTopo->tobStereo(detid.rawId());
      break;
    }
  case StripSubdetector::TEC:
    {
      
      // is this module in TEC+ or TEC-?
      side = tTopo->tecSide(detid.rawId());
      detector = "TEC";
      layer = tTopo->tecWheel(detid.rawId());
      stereo = tTopo->tecStereo(detid.rawId());
      break;
    }
  case StripSubdetector::TID:
    {
      
      // is this module in TID+ or TID-?
      side = tTopo->tidSide(detid.rawId());
      detector = "TID";
      layer = tTopo->tidWheel(detid.rawId());
      stereo = tTopo->tidStereo(detid.rawId());
      break;
    }
  }
  std::string name( detector + "\t" + boost::lexical_cast<string>(layer) + "\t" + boost::lexical_cast<string>(stereo) + "\t" );
  if( side == 1 ) {
    name += "-";
  }
  else if ( side == 2 ) {
    name += "+";
  }
//   if( side != -1 ) {
//     name += boost::lexical_cast<string>(side);
//   }

  return name;
}
