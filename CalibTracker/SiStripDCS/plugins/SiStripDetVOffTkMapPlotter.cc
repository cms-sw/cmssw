#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <sstream>

#include "CondCore/CondDB/interface/ConnectionPool.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/Common/interface/TimeConversions.h"

#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"


class SiStripDetVOffTkMapPlotter : public edm::EDAnalyzer {
public:
  explicit SiStripDetVOffTkMapPlotter(const edm::ParameterSet& iConfig );
  virtual ~SiStripDetVOffTkMapPlotter();
  virtual void analyze( const edm::Event& evt, const edm::EventSetup& evtSetup);
  virtual void endJob();

private:
  std::string formatIOV(cond::Time_t iov, std::string format="%Y-%m-%d__%H_%M_%S");

  cond::persistency::ConnectionPool m_connectionPool;
  std::string m_condDb;
  std::string m_plotTag;

  // IOV of plotting.
  cond::Time_t m_IOV;
  // Or use datatime string. Format: "2002-01-20 23:59:59.000". Set IOV to 0 to use this.
  std::string m_Time;
  // Set the plot format. Default: png.
  std::string m_plotFormat;
  // Specify output root file name. Leave empty if do not want to save plots in a root file.
  std::string m_outputFile;

  edm::Service<SiStripDetInfoFileReader> detidReader;
};

SiStripDetVOffTkMapPlotter::SiStripDetVOffTkMapPlotter(const edm::ParameterSet& iConfig):
    m_connectionPool(),
    m_condDb( iConfig.getParameter< std::string >("conditionDatabase") ),
    m_plotTag( iConfig.getParameter< std::string >("Tag") ),
    m_IOV( iConfig.getUntrackedParameter< cond::Time_t >("IOV", 0) ),
    m_Time( iConfig.getUntrackedParameter< std::string >("Time", "") ),
    m_plotFormat( iConfig.getUntrackedParameter< std::string >("plotFormat", "png") ),
    m_outputFile( iConfig.getUntrackedParameter< std::string >("outputFile", "") ){
  m_connectionPool.setParameters( iConfig.getParameter<edm::ParameterSet>("DBParameters")  );
  m_connectionPool.configure();
}

SiStripDetVOffTkMapPlotter::~SiStripDetVOffTkMapPlotter() {
}

void SiStripDetVOffTkMapPlotter::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {

  cond::Time_t theIov = 0;
  if (m_IOV != 0){
    theIov = m_IOV;
  }else if (!m_Time.empty()){
    theIov = cond::time::from_boost( boost::posix_time::time_from_string(m_Time) );
  }else{
    // Use the current time if no input. Will get the last IOV.
    theIov = cond::time::from_boost( boost::posix_time::second_clock::universal_time() );
  }

  // open db session
  edm::LogInfo("SiStripDetVOffMapPlotter") << "[SiStripDetVOffMapPlotter::" << __func__ << "] "
      << "Query the condition database " << m_condDb << " for tag " << m_plotTag;
  cond::persistency::Session condDbSession = m_connectionPool.createSession( m_condDb );
  condDbSession.transaction().start( true );
  cond::persistency::IOVProxy iovProxy = condDbSession.readIov(m_plotTag, true);
  auto iiov = iovProxy.find(theIov);
  if (iiov==iovProxy.end())
    throw cms::Exception("Input IOV "+std::to_string(m_IOV)+"/"+m_Time+" is invalid!");

  theIov = (*iiov).since;
  edm::LogInfo("SiStripDetVOffMapPlotter") << "[SiStripDetVOffMapPlotter::" << __func__ << "] "
      << "Make tkMap for IOV " << theIov << " (" << boost::posix_time::to_simple_string(cond::time::to_boost(theIov)) << ")";
  auto payload = condDbSession.fetchPayload<SiStripDetVOff>( (*iiov).payloadId );

  TrackerMap lvmap,hvmap;
  TkHistoMap lvhisto("LV_Status","LV_Status",-1);
  TkHistoMap hvhisto("HV_Status","HV_Status",-1);

  auto detids = detidReader->getAllDetIds();
  for (auto id : detids){
    if (payload->IsModuleLVOff(id))
      lvhisto.fill(id, 1); // RED
    else
      lvhisto.fill(id, 0.5);

    if (payload->IsModuleHVOff(id))
      hvhisto.fill(id, 1); // RED
    else
      hvhisto.fill(id, 0.5);
  }

  lvhisto.dumpInTkMap(&lvmap);
  hvhisto.dumpInTkMap(&hvmap);
  lvmap.setPalette(1);
  hvmap.setPalette(1);
  lvmap.save(true,0,0,"LV_tkMap_"+formatIOV(theIov)+"."+m_plotFormat);
  hvmap.save(true,0,0,"HV_tkMap_"+formatIOV(theIov)+"."+m_plotFormat);

  if (!m_outputFile.empty()){
    lvhisto.save(m_outputFile);
    hvhisto.save(m_outputFile);
  }

}

void SiStripDetVOffTkMapPlotter::endJob() {
}

std::string SiStripDetVOffTkMapPlotter::formatIOV(cond::Time_t iov, std::string format) {
  auto facet = new boost::posix_time::time_facet(format.c_str());
   std::ostringstream stream;
   stream.imbue(std::locale(stream.getloc(), facet));
   stream << cond::time::to_boost(iov);
   return stream.str();
}

DEFINE_FWK_MODULE(SiStripDetVOffTkMapPlotter);

