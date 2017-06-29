#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DQMOffline/CalibTracker/plugins/SiStripPopConHistoryDQMBase.h"

/**
 * @class SiStripPopConHistoryDQM
 * @author D. Giordano, A.-C. Le Bihan
 *
 * @EDAnalyzer to read DQM root file & insert summary informations to DB
 */
class SiStripPopConHistoryDQM : public SiStripPopConHistoryDQMBase
{
public:
  explicit SiStripPopConHistoryDQM(const edm::ParameterSet& pset)
    : SiStripPopConHistoryDQMBase(pset)
  {}

  void init(const edm::EventSetup&);

  virtual ~SiStripPopConHistoryDQM();
private:
  uint32_t returnDetComponent(const MonitorElement* ME) const;
  bool setDBLabelsForUser(const std::string& keyName, std::vector<std::string>& userDBContent, const std::string& quantity) const;
  bool setDBValuesForUser(const MonitorElement* me, HDQMSummary::InputVector& values, const std::string& quantity) const;
private:
  const TrackerTopology* trackerTopo_;
};

SiStripPopConHistoryDQM::~SiStripPopConHistoryDQM() {}

void SiStripPopConHistoryDQM::init(const edm::EventSetup& setup)
{
  edm::ESHandle<TrackerTopology> tTopo;
  setup.get<TrackerTopologyRcd>().get(tTopo);
  trackerTopo_ = tTopo.product();
}

uint32_t SiStripPopConHistoryDQM::returnDetComponent(const MonitorElement* ME) const
{
  LogTrace("SiStripHistoryDQMService") <<  "[SiStripHistoryDQMService::returnDetComponent]";
  const std::string str{ME->getName()};
  const size_t __key_length__=7;
  const size_t __detid_length__=9;

  uint32_t layer=0,side=0;

  if ( str.find("__det__") != std::string::npos ) {
    return atoi(str.substr(str.find("__det__")+__key_length__,__detid_length__).c_str());
  }
  //TIB
  else if ( str.find("TIB") != std::string::npos ) {
    if ( str.find("layer")!= std::string::npos )
      layer = atoi(str.substr(str.find("layer__")+__key_length__,1).c_str());
    return trackerTopo_->tibDetId(layer,0,0,0,0,0).rawId();
  }
  //TOB
  else if ( str.find("TOB") != std::string::npos ) {
    if ( str.find("layer") != std::string::npos )
      layer = atoi(str.substr(str.find("layer__")+__key_length__,1).c_str());
    return trackerTopo_->tobDetId(layer,0,0,0,0).rawId();
  }
  //TID
  else if ( str.find("TID") != std::string::npos ) {
    if ( str.find("side") != std::string::npos ) {
      side = atoi(str.substr(str.find("_side__")+__key_length__,1).c_str());
      if ( str.find("wheel") != std::string::npos ) {
	layer = atoi(str.substr(str.find("wheel__")+__key_length__,1).c_str());
      }
    }
    return trackerTopo_->tidDetId(side,layer,0,0,0,0).rawId();
  }
  //TEC
  else if ( str.find("TEC") != std::string::npos ) {
    if ( str.find("side") != std::string::npos ) {
      side = atoi(str.substr(str.find("_side__")+__key_length__,1).c_str());
      if ( str.find("wheel") != std::string::npos ) {
	layer = atoi(str.substr(str.find("wheel__")+__key_length__,1).c_str());
      }
    }
    return trackerTopo_->tecDetId(side,layer,0,0,0,0,0).rawId();
  }
  else
    return DetId(DetId::Tracker,0).rawId(); //Full Tracker
}

//Example on how to define an user function for the statistic extraction
bool SiStripPopConHistoryDQM::setDBLabelsForUser(const std::string& keyName, std::vector<std::string>& userDBContent, const std::string& quantity) const
{
  if (quantity == "user_2DYmean") {
    userDBContent.push_back(keyName+sep()+std::string("yMean"));
    userDBContent.push_back(keyName+sep()+std::string("yError"));
  } else {
    edm::LogError("SiStripHistoryDQMService") << "ERROR: quantity does not exist in SiStripHistoryDQMService::setDBValuesForUser(): " << quantity;
    return false;
  }
  return true;
}

bool SiStripPopConHistoryDQM::setDBValuesForUser(const MonitorElement* me, HDQMSummary::InputVector& values, const std::string& quantity) const
{
  if (quantity == "user_2DYmean") {
    TH2F* Hist = (TH2F*) me->getTH2F();
    values.push_back( Hist->GetMean(2) );
    values.push_back( Hist->GetRMS(2) );
  } else {
    edm::LogError("SiStripHistoryDQMService") << "ERROR: quantity does not exist in SiStripHistoryDQMService::setDBValuesForUser(): " << quantity;
    return false;
  }
  return true;
}

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CondCore/PopCon/interface/PopCon.h"

// copied from popCon::PopConAnalyzer
// modified to pass an edm::EventSetup reference at begin run
class SiStripDQMHistoryPopCon : public edm::EDAnalyzer
{
public:
  using SourceHandler = SiStripPopConHistoryDQM;

  SiStripDQMHistoryPopCon(const edm::ParameterSet& pset) :
    m_populator(pset),
    m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

  virtual ~SiStripDQMHistoryPopCon() {}

private:
  virtual void beginJob() {}
  virtual void endJob  () {
    write();
  }

  virtual void beginRun(const edm::Run&, const edm::EventSetup& setup)
  {
    m_source.init(setup);
  }

  virtual void analyze(const edm::Event&, const edm::EventSetup&) {}

  void write() {
    m_populator.write(m_source);
  }
private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripDQMHistoryPopCon);
