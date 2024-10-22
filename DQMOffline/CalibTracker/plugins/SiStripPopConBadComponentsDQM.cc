#include "DQMOffline/CalibTracker/plugins/SiStripDQMPopConSourceHandler.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

/**
  @class SiStripBadComponentsDQMService
  @author M. De Mattia, S. Dutta, D. Giordano

  @popcon::PopConSourceHandler to read modules flagged by the DQM as bad and write in the database.
*/
class SiStripPopConBadComponentsHandlerFromDQM : public SiStripDQMPopConSourceHandler<SiStripBadStrip> {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

  explicit SiStripPopConBadComponentsHandlerFromDQM(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);
  ~SiStripPopConBadComponentsHandlerFromDQM() override;
  // interface methods: implemented in template
  void initES(const edm::EventSetup&) override;
  void dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) override;
  SiStripBadStrip* getObj() const override;

protected:
  std::string getMetaDataString() const override;

private:
  edm::FileInPath fp_;
  SiStripBadStrip m_obj;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const TrackerTopology* trackerTopo_;
};

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

SiStripPopConBadComponentsHandlerFromDQM::SiStripPopConBadComponentsHandlerFromDQM(const edm::ParameterSet& iConfig,
                                                                                   edm::ConsumesCollector&& iC)
    : SiStripDQMPopConSourceHandler<SiStripBadStrip>(iConfig),
      fp_{iConfig.getUntrackedParameter<edm::FileInPath>("file",
                                                         edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile))},
      tTopoToken_(iC.esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
  edm::LogInfo("SiStripBadComponentsDQMService") << "[SiStripBadComponentsDQMService::SiStripBadComponentsDQMService]";
}

SiStripPopConBadComponentsHandlerFromDQM::~SiStripPopConBadComponentsHandlerFromDQM() {
  edm::LogInfo("SiStripBadComponentsDQMService") << "[SiStripBadComponentsDQMService::~SiStripBadComponentsDQMService]";
}

void SiStripPopConBadComponentsHandlerFromDQM::initES(const edm::EventSetup& setup) {
  trackerTopo_ = &setup.getData(tTopoToken_);
}

std::string SiStripPopConBadComponentsHandlerFromDQM::getMetaDataString() const {
  std::stringstream ss;
  ss << SiStripDQMPopConSourceHandler<SiStripBadStrip>::getMetaDataString();
  getObj()->printSummary(ss, trackerTopo_);
  return ss.str();
}

namespace {
  void getModuleFolderList(SiStripPopConBadComponentsHandlerFromDQM::DQMStore::IGetter& getter,
                           const std::string& pwd,
                           std::vector<std::string>& mfolders) {
    if (std::string::npos != pwd.find("module_")) {
      //    std::string mId = pwd.substr(pwd.find("module_")+7, 9);
      mfolders.push_back(pwd);
    } else {
      for (const auto& subdir : getter.getSubdirs()) {
        getter.cd(subdir);
        getModuleFolderList(getter, subdir, mfolders);
        getter.cd();
        getter.cd(pwd);
      }
    }
  }
}  // namespace

void SiStripPopConBadComponentsHandlerFromDQM::dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter& getter) {
  //*LOOP OVER THE LIST OF SUMMARY OBJECTS TO INSERT IN DB*//

  m_obj = SiStripBadStrip();

  const auto detInfo = SiStripDetInfoFileReader::read(fp_.fullPath());

  getter.cd();

  const std::string mechanicalview_dir = "MechanicalView";
  if (!getter.dirExists(mechanicalview_dir))
    return;

  const std::vector<std::string> subdet_folder = {"TIB", "TOB", "TEC/side_1", "TEC/side_2", "TID/side_1", "TID/side_2"};

  for (const auto& im : subdet_folder) {
    const std::string dname = mechanicalview_dir + "/" + im;
    getter.cd();
    if (!getter.dirExists(dname))
      continue;
    getter.cd(dname);

    std::vector<std::string> module_folders;
    getModuleFolderList(getter, dname, module_folders);

    const std::string bad_module_folder = dname + "/" + "BadModuleList";
    getter.cd();
    if (getter.dirExists(bad_module_folder)) {
      for (const MonitorElement* me : getter.getContents(bad_module_folder)) {
        std::cout << me->getName() << " " << me->getIntValue() << std::endl;
        uint32_t detId = std::stoul(me->getName());
        short flag = me->getIntValue();

        std::vector<unsigned int> theSiStripVector;

        unsigned short firstBadStrip = 0, NconsecutiveBadStrips = 0;
        unsigned int theBadStripRange;

        // for(std::vector<uint32_t>::const_iterator is=BadApvList_.begin(); is!=BadApvList_.end(); ++is){

        //   firstBadStrip=(*is)*128;
        NconsecutiveBadStrips = detInfo.getNumberOfApvsAndStripLength(detId).first * 128;

        theBadStripRange = m_obj.encode(firstBadStrip, NconsecutiveBadStrips, flag);

        LogDebug("SiStripBadComponentsDQMService") << "detid " << detId << " \t"
                                                   << ", flag " << flag << std::endl;

        theSiStripVector.push_back(theBadStripRange);
        // }

        SiStripBadStrip::Range range(theSiStripVector.begin(), theSiStripVector.end());
        if (!m_obj.put(detId, range)) {
          edm::LogError("SiStripBadFiberBuilder")
              << "[SiStripBadFiberBuilder::analyze] detid already exists" << std::endl;
        }
      }
    }
  }
  getter.cd();
}

SiStripBadStrip* SiStripPopConBadComponentsHandlerFromDQM::getObj() const { return new SiStripBadStrip(m_obj); }

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMOffline/CalibTracker/plugins/SiStripPopConDQMEDHarvester.h"
using SiStripPopConBadComponentsDQM = SiStripPopConDQMEDHarvester<SiStripPopConBadComponentsHandlerFromDQM>;
DEFINE_FWK_MODULE(SiStripPopConBadComponentsDQM);
