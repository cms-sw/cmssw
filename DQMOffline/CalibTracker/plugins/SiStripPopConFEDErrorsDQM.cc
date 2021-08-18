#include "FWCore/Framework/interface/ESWatcher.h"
#include "DQMOffline/CalibTracker/plugins/SiStripDQMPopConSourceHandler.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
class FedChannelConnection;
class SiStripFedCabling;

/**
  @class SiStripPopConFEDErrorsHandlerFromDQM
  @author A.-M. Magnan, M. De Mattia
  @author P. David update to PopConSourceHandler and DQMEDHarvester
  @EDAnalyzer to read modules flagged by the DQM due to FED errors as bad and write in the database with the proper error flag.
*/
class SiStripPopConFEDErrorsHandlerFromDQM : public SiStripDQMPopConSourceHandler<SiStripBadStrip> {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

  explicit SiStripPopConFEDErrorsHandlerFromDQM(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);
  ~SiStripPopConFEDErrorsHandlerFromDQM() override;
  // interface methods: implemented in template
  void initES(const edm::EventSetup& iSetup) override;
  void dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) override;
  SiStripBadStrip* getObj() const override;

private:
  void readHistogram(MonitorElement* aMe, unsigned int& aCounter, const float aNorm, const unsigned int aFedId);

  void addBadAPV(const FedChannelConnection& aConnection,
                 const unsigned short aAPVNumber,
                 const unsigned short aFlag,
                 unsigned int& aCounter);

  void addBadStrips(const FedChannelConnection& aConnection,
                    const unsigned int aDetId,
                    const unsigned short aApvNum,
                    const unsigned short aFlag,
                    unsigned int& aCounter);

  /// Writes the errors to the db
  void addErrors();

private:
  double threshold_;
  unsigned int debug_;
  edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> fedCablingToken_;
  edm::ESWatcher<SiStripFedCablingRcd> fedCablingWatcher_;
  const SiStripFedCabling* cabling_;
  SiStripBadStrip obj_;
  std::map<uint32_t, std::vector<unsigned int> > detIdErrors_;
};

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DQMServices/Core/interface/DQMStore.h"

SiStripPopConFEDErrorsHandlerFromDQM::SiStripPopConFEDErrorsHandlerFromDQM(const edm::ParameterSet& iConfig,
                                                                           edm::ConsumesCollector&& iC)
    : SiStripDQMPopConSourceHandler<SiStripBadStrip>(iConfig),
      threshold_(iConfig.getUntrackedParameter<double>("Threshold", 0)),
      debug_(iConfig.getUntrackedParameter<unsigned int>("Debug", 0)),
      fedCablingToken_(iC.esConsumes<SiStripFedCabling, SiStripFedCablingRcd, edm::Transition::BeginRun>()) {
  edm::LogInfo("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::SiStripFEDErrorsDQM()]";
}

SiStripPopConFEDErrorsHandlerFromDQM::~SiStripPopConFEDErrorsHandlerFromDQM() {
  edm::LogInfo("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::~SiStripFEDErrorsDQM]";
}

void SiStripPopConFEDErrorsHandlerFromDQM::initES(const edm::EventSetup& iSetup) {
  if (fedCablingWatcher_.check(iSetup)) {
    cabling_ = &iSetup.getData(fedCablingToken_);
  }
}

void SiStripPopConFEDErrorsHandlerFromDQM::dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter& getter) {
  obj_ = SiStripBadStrip();

  std::ostringstream lPath;
  lPath << "Run " << getRunNumber() << "/SiStrip/Run summary/ReadoutView/";
  const std::string lBaseDir = lPath.str();

  getter.setCurrentFolder(lBaseDir);
  LogTrace("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readBadAPVs] Now in " << lBaseDir << std::endl;

  std::vector<std::pair<std::string, unsigned int> > lFedsFolder;
  //for FED errors, use summary folder and fedId=0
  //do not put a slash or "goToDir" won't work...
  lFedsFolder.push_back(std::pair<std::string, unsigned int>("FedMonitoringSummary", 0));

  //for FE/channel/APV errors, they are written in a folder per FED,
  //if there was at least one error.
  //So just loop on folders and see which ones exist.
  for (unsigned int ifed(FEDNumbering::MINSiStripFEDID); ifed <= FEDNumbering::MAXSiStripFEDID;
       ifed++) {  //loop on FEDs

    std::ostringstream lFedDir;
    lFedDir << "FrontEndDriver" << ifed;
    if (!getter.dirExists(lFedDir.str()))
      continue;
    else {
      if (debug_)
        LogTrace("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readBadAPVs] - Errors detected for FED " << ifed
                                        << std::endl;
      lFedsFolder.push_back(std::pair<std::string, unsigned int>(lFedDir.str(), ifed));
    }
  }
  getter.cd();

  unsigned int nAPVsTotal = 0;
  //retrieve total number of APVs valid and connected from cabling:
  if (!cabling_) {
    edm::LogError("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readBadAPVs] cabling not filled, return false "
                                         << std::endl;
    return;
  }
  auto lFedVec = cabling_->fedIds();
  for (unsigned int iFed(0); iFed < lFedVec.size(); iFed++) {
    if (*(lFedVec.begin() + iFed) < sistrip::FED_ID_MIN || *(lFedVec.begin() + iFed) > sistrip::FED_ID_MAX) {
      edm::LogError("SiStripFEDErrorsDQM")
          << "[SiStripFEDErrorsDQM::readBadAPVs] Invalid fedid : " << *(lFedVec.begin() + iFed) << std::endl;
      continue;
    }
    auto lConnVec = cabling_->fedConnections(*(lFedVec.begin() + iFed));
    for (unsigned int iConn(0); iConn < lConnVec.size(); iConn++) {
      const FedChannelConnection& lConnection = *(lConnVec.begin() + iConn);
      if (!lConnection.isConnected())
        continue;
      unsigned int lDetid = lConnection.detId();
      if (!lDetid || lDetid == sistrip::invalid32_)
        continue;
      //2 APVs per channel....
      nAPVsTotal += 2;
    }
  }

  edm::LogInfo("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readBadAPVs] Total number of APVs found : "
                                      << nAPVsTotal << std::endl;

  unsigned int nAPVsWithErrorTotal = 0;
  unsigned int nFolders = 0;
  float lNorm = 0;
  for (const auto& iFolder : lFedsFolder) {
    const std::string lDirName = lBaseDir + "/" + iFolder.first;
    const unsigned int lFedId = iFolder.second;

    if (!getter.dirExists(lDirName))
      continue;

    std::vector<MonitorElement*> lMeVec = getter.getContents(lDirName);

    if (nFolders == 0) {
      for (auto iMe : lMeVec) {  //loop on ME found in directory
        std::string lMeName = iMe->getName();
        if (lMeName.find("nFEDErrors") != lMeName.npos) {
          lNorm = iMe->getEntries();
        }
      }
      //if norm histo has not been found, no point in continuing....
      if (lNorm < 1) {
        edm::LogError("SiStripFEDErrorsDQM")
            << "[SiStripFEDErrorsDQM::readBadAPVs] nFEDErrors not found, norm is " << lNorm << std::endl;
        return;
      }
    }

    unsigned int nAPVsWithError = 0;
    for (auto iMe : lMeVec) {  //loop on ME found in directory
      if (iMe->getEntries() == 0)
        continue;
      const std::string lMeName = iMe->getName();

      bool lookForErrors = false;
      if (nFolders == 0) {
        //for the first element of lFedsFolder: this is FED errors
        lookForErrors = lMeName.find("DataMissing") != lMeName.npos || lMeName.find("AnyFEDErrors") != lMeName.npos ||
                        (lMeName.find("CorruptBuffer") != lMeName.npos && lMeName.find("nFED") == lMeName.npos);
      } else {
        //for the others, it is channel or FE errors.
        lookForErrors = lMeName.find("APVAddressError") != lMeName.npos || lMeName.find("APVError") != lMeName.npos ||
                        lMeName.find("BadMajorityAddresses") != lMeName.npos ||
                        lMeName.find("FEMissing") != lMeName.npos || lMeName.find("OOSBits") != lMeName.npos ||
                        lMeName.find("UnlockedBits") != lMeName.npos;
      }

      if (lookForErrors)
        readHistogram(iMe, nAPVsWithError, lNorm, lFedId);

    }  //loop on ME found in directory

    nAPVsWithErrorTotal += nAPVsWithError;
    ++nFolders;
  }  //loop on lFedsFolders

  edm::LogInfo("SiStripFEDErrorsDQM")
      << "[SiStripFEDErrorsDQM::readBadAPVs] Total APVs with error found above threshold = " << nAPVsWithErrorTotal
      << std::endl;

  getter.cd();

  addErrors();
}

SiStripBadStrip* SiStripPopConFEDErrorsHandlerFromDQM::getObj() const { return new SiStripBadStrip(obj_); }

void SiStripPopConFEDErrorsHandlerFromDQM::readHistogram(MonitorElement* aMe,
                                                         unsigned int& aCounter,
                                                         const float aNorm,
                                                         const unsigned int aFedId) {
  unsigned short lFlag = 0;
  std::string lMeName = aMe->getName();
  if (lMeName.find("DataMissing") != lMeName.npos) {
    lFlag = 0;
  } else if (lMeName.find("AnyFEDErrors") != lMeName.npos) {
    lFlag = 1;
  } else if (lMeName.find("CorruptBuffer") != lMeName.npos && lMeName.find("nFED") == lMeName.npos) {
    lFlag = 2;
  } else if (lMeName.find("FEMissing") != lMeName.npos) {
    lFlag = 3;
  } else if (lMeName.find("BadMajorityAddresses") != lMeName.npos) {
    lFlag = 4;
  } else if (lMeName.find("UnlockedBits") != lMeName.npos) {
    lFlag = 5;
  } else if (lMeName.find("OOSBits") != lMeName.npos) {
    lFlag = 6;
  } else if (lMeName.find("APVAddressError") != lMeName.npos) {
    lFlag = 7;
  } else if (lMeName.find("APVError") != lMeName.npos) {
    lFlag = 8;
  } else {
    edm::LogError("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readHistogramError] Shouldn't be here ..."
                                         << std::endl;
    return;
  }

  if (debug_) {
    LogTrace("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readHistogramError] Reading histo : " << lMeName
                                    << ", flag = " << lFlag << std::endl;
  }

  unsigned int lNBins = aMe->getNbinsX();
  int lBinShift = 0;
  bool lIsFedHist = false;
  bool lIsAPVHist = false;
  bool lIsFeHist = false;
  bool lIsChHist = false;

  if (lNBins > 200) {
    lBinShift = FEDNumbering::MINSiStripFEDID - 1;  //shift for FED ID from bin number
    lIsFedHist = true;
  } else {
    lBinShift = -1;  //shift for channel/APV/FE id from bin number
    if (lNBins > 100)
      lIsAPVHist = true;
    else if (lNBins < 10)
      lIsFeHist = true;
    else
      lIsChHist = true;
  }

  if (debug_) {
    LogTrace("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::readHistogramError] lIsFedHist: " << lIsFedHist
                                    << std::endl
                                    << "[SiStripFEDErrorsDQM::readHistogramError] lIsAPVHist: " << lIsAPVHist
                                    << std::endl
                                    << "[SiStripFEDErrorsDQM::readHistogramError] lIsFeHist : " << lIsFeHist
                                    << std::endl
                                    << "[SiStripFEDErrorsDQM::readHistogramError] lIsChHist : " << lIsChHist
                                    << std::endl;
  }

  for (unsigned int ibin(1); ibin < lNBins + 1; ibin++) {
    if (aMe->getBinContent(ibin) > 0) {
      float lStat = aMe->getBinContent(ibin) * 1. / aNorm;
      if (lStat <= threshold_) {
        if (debug_)
          LogTrace("SiStripFEDErrorsDQM")
              << "[SiStripFEDErrorsDQM::readHistogramError] ---- Below threshold : " << lStat << std::endl;
        continue;
      }
      if (lIsFedHist) {
        unsigned int lFedId = ibin + lBinShift;
        //loop on all enabled channels of this FED....
        for (unsigned int iChId = 0; iChId < sistrip::FEDCH_PER_FED; iChId++) {  //loop on channels
          const FedChannelConnection& lConnection = cabling_->fedConnection(lFedId, iChId);
          if (!lConnection.isConnected())
            continue;
          addBadAPV(lConnection, 0, lFlag, aCounter);
        }
      } else {
        if (lIsFeHist) {
          unsigned int iFeId = ibin + lBinShift;
          //loop on all enabled channels of this FE....
          for (unsigned int iFeCh = 0; iFeCh < sistrip::FEDCH_PER_FEUNIT; iFeCh++) {  //loop on channels
            unsigned int iChId = sistrip::FEDCH_PER_FEUNIT * iFeId + iFeCh;
            const FedChannelConnection& lConnection = cabling_->fedConnection(aFedId, iChId);
            if (!lConnection.isConnected())
              continue;
            addBadAPV(lConnection, 0, lFlag, aCounter);
          }
        } else {
          unsigned int iChId = ibin + lBinShift;
          if (lIsAPVHist) {
            unsigned int iAPVid = iChId % 2 + 1;
            iChId = static_cast<unsigned int>(iChId / 2.);
            const FedChannelConnection& lConnection = cabling_->fedConnection(aFedId, iChId);
            addBadAPV(lConnection, iAPVid, lFlag, aCounter);

          }  //ifAPVhists
          else {
            const FedChannelConnection& lConnection = cabling_->fedConnection(aFedId, iChId);
            addBadAPV(lConnection, 0, lFlag, aCounter);
          }
        }  //if not FE hist
      }    //if not FED hist
    }      //if entries in histo
  }        //loop on bins
}  //method readHistogram

void SiStripPopConFEDErrorsHandlerFromDQM::addBadAPV(const FedChannelConnection& aConnection,
                                                     const unsigned short aAPVNumber,
                                                     const unsigned short aFlag,
                                                     unsigned int& aCounter) {
  if (!aConnection.isConnected()) {
    edm::LogWarning("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::addBadAPV] Warning, incompatible cabling ! "
                                              "Channel is not connected, but entry found in histo ... "
                                           << std::endl;
    return;
  }
  unsigned int lDetid = aConnection.detId();
  if (!lDetid || lDetid == sistrip::invalid32_) {
    edm::LogWarning("SiStripFEDErrorsDQM")
        << "[SiStripFEDErrorsDQM::addBadAPV] Warning, DetId is invalid: " << lDetid << std::endl;
    return;
  }
  //unsigned short nChInModule = aConnection.nApvPairs();
  unsigned short lApvNum = 0;
  if (aAPVNumber < 2) {
    lApvNum = 2 * aConnection.apvPairNumber();
    addBadStrips(aConnection, lDetid, lApvNum, aFlag, aCounter);
  }
  if (aAPVNumber == 0 || aAPVNumber == 2) {
    lApvNum = 2 * aConnection.apvPairNumber() + 1;
    addBadStrips(aConnection, lDetid, lApvNum, aFlag, aCounter);
  }
}

void SiStripPopConFEDErrorsHandlerFromDQM::addBadStrips(const FedChannelConnection& aConnection,
                                                        const unsigned int aDetId,
                                                        const unsigned short aApvNum,
                                                        const unsigned short aFlag,
                                                        unsigned int& aCounter) {
  // std::vector<unsigned int> lStripVector;
  const unsigned short lFirstBadStrip = aApvNum * 128;
  const unsigned short lConsecutiveBadStrips = 128;

  unsigned int lBadStripRange = obj_.encode(lFirstBadStrip, lConsecutiveBadStrips, aFlag);

  LogTrace("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::addBadStrips] ---- Adding : detid " << aDetId << " (FED "
                                  << aConnection.fedId() << ", Ch " << aConnection.fedCh() << ")"
                                  << ", APV " << aApvNum << ", flag " << aFlag << std::endl;

  detIdErrors_[aDetId].push_back(lBadStripRange);

  // lStripVector.push_back(lBadStripRange);
  // SiStripBadStrip::Range lRange(lStripVector.begin(),lStripVector.end());
  // if ( !obj.put(aDetId,lRange) ) {
  //   edm::LogError("SiStripFEDErrorsDQM")<<"[SiStripFEDErrorsDQM::addBadStrips] detid already exists." << std::endl;
  // }

  aCounter++;
}

namespace {
  // set corresponding bit to 1 in flag
  inline void setFlagBit(unsigned short& aFlag, const unsigned short aBit) { aFlag = aFlag | (0x1 << aBit); }
}  // namespace

void SiStripPopConFEDErrorsHandlerFromDQM::addErrors() {
  for (const auto& it : detIdErrors_) {
    const std::vector<uint32_t>& lList = it.second;

    //map of first strip number and flag
    //purpose is to encode all existing flags into a unique one...
    std::map<unsigned short, unsigned short> lAPVMap;
    for (auto cCont : lList) {
      SiStripBadStrip::data lData = obj_.decode(cCont);
      unsigned short lFlag = 0;
      setFlagBit(lFlag, lData.flag);

      //std::cout << " -- Detid " << it.first << ", strip " << lData.firstStrip << ", flag " << lData.flag << std::endl;

      auto lInsert = lAPVMap.emplace(lData.firstStrip, lFlag);
      if (!lInsert.second) {
        //std::cout << " ---- Adding bit : " << lData.flag << " to " << lInsert.first->second << ": ";
        setFlagBit(lInsert.first->second, lData.flag);
        //std::cout << lInsert.first->second << std::endl;
      }
    }

    //encode the new flag
    std::vector<unsigned int> lStripVector;
    const unsigned short lConsecutiveBadStrips = 128;
    lStripVector.reserve(lAPVMap.size());
    for (const auto& lIter : lAPVMap) {
      lStripVector.push_back(obj_.encode(lIter.first, lConsecutiveBadStrips, lIter.second));
    }

    SiStripBadStrip::Range lRange(lStripVector.begin(), lStripVector.end());
    if (!obj_.put(it.first, lRange)) {
      edm::LogError("SiStripFEDErrorsDQM") << "[SiStripFEDErrorsDQM::addBadStrips] detid already exists." << std::endl;
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMOffline/CalibTracker/plugins/SiStripPopConDQMEDHarvester.h"
using SiStripPopConFEDErrorsDQM = SiStripPopConDQMEDHarvester<SiStripPopConFEDErrorsHandlerFromDQM>;
DEFINE_FWK_MODULE(SiStripPopConFEDErrorsDQM);
