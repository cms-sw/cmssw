#include "DQM/EcalMonitorTasks/interface/RawDataTask.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/FEFlags.h"

#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"

namespace ecaldqm {

  RawDataTask::RawDataTask()
      : DQWorkerTask(), runNumber_(0), l1A_(0), orbit_(0), bx_(0), triggerType_(0), feL1Offset_(0) {}

  void RawDataTask::addDependencies(DependencySet& _dependencies) {
    _dependencies.push_back(Dependency(kEcalRawData, kSource));
  }

  void RawDataTask::beginRun(edm::Run const& _run, edm::EventSetup const&) { runNumber_ = _run.run(); }

  void RawDataTask::beginEvent(edm::Event const& _evt, edm::EventSetup const&, bool const& ByLumiResetSwitch, bool&) {
    orbit_ = _evt.orbitNumber() & 0xffffffff;
    bx_ = _evt.bunchCrossing() & 0xfff;
    triggerType_ = _evt.experimentType() & 0xf;
    l1A_ = 0;
    feL1Offset_ = _evt.isRealData() ? 1 : 0;
    if (ByLumiResetSwitch) {
      MEs_.at("DesyncByLumi").reset(GetElectronicsMap());
      MEs_.at("FEByLumi").reset(GetElectronicsMap());
      MEs_.at("FEStatusErrMapByLumi").reset(GetElectronicsMap());
    }
  }

  void RawDataTask::runOnSource(FEDRawDataCollection const& _fedRaw) {
    MESet& meCRC(MEs_.at("CRC"));

    // Get GT L1 info
    const FEDRawData& gtFED(_fedRaw.FEDData(812));
    if (gtFED.size() > sizeof(uint64_t)) {  // FED header is one 64 bit word
      const uint32_t* halfHeader = reinterpret_cast<const uint32_t*>(gtFED.data());
      l1A_ = *(halfHeader + 1) & 0xffffff;
    }

    for (int iFED(601); iFED <= 654; iFED++) {
      const FEDRawData& fedData(_fedRaw.FEDData(iFED));
      unsigned length(fedData.size() / sizeof(uint64_t));
      if (length > 1) {  // FED header is one 64 bit word
        const uint64_t* pData(reinterpret_cast<uint64_t const*>(fedData.data()));
        if ((pData[length - 1] & 0x4) != 0)
          meCRC.fill(getEcalDQMSetupObjects(), iFED - 600);
      }
    }
  }

  void RawDataTask::runOnRawData(EcalRawDataCollection const& _dcchs) {
    using namespace std;

    MESet& meRunNumber(MEs_.at("RunNumber"));
    MESet& meOrbit(MEs_.at("Orbit"));
    MESet& meOrbitDiff(MEs_.at("OrbitDiff"));
    MESet& meTriggerType(MEs_.at("TriggerType"));
    MESet& meL1ADCC(MEs_.at("L1ADCC"));
    MESet& meBXDCC(MEs_.at("BXDCC"));
    MESet& meBXDCCDiff(MEs_.at("BXDCCDiff"));
    MESet& meBXFE(MEs_.at("BXFE"));
    MESet& meBXFEDiff(MEs_.at("BXFEDiff"));
    MESet& meBXFEInvalid(MEs_.at("BXFEInvalid"));
    MESet& meL1AFE(MEs_.at("L1AFE"));
    MESet& meFEStatus(MEs_.at("FEStatus"));
    MESet& meFEStatusErrMapByLumi(MEs_.at("FEStatusErrMapByLumi"));
    MESet& meFEStatusMEM(MEs_.at("FEStatusMEM"));
    MESet& meDesyncByLumi(MEs_.at("DesyncByLumi"));
    MESet& meDesyncTotal(MEs_.at("DesyncTotal"));
    MESet& meFEByLumi(MEs_.at("FEByLumi"));
    MESet& meBXTCC(MEs_.at("BXTCC"));
    MESet& meL1ATCC(MEs_.at("L1ATCC"));
    MESet& meBXSRP(MEs_.at("BXSRP"));
    MESet& meL1ASRP(MEs_.at("L1ASRP"));
    MESet& meTrendNSyncErrors(MEs_.at("L1ATCC"));
    MESet& meEventTypePreCalib(MEs_.at("EventTypePreCalib"));
    MESet& meEventTypeCalib(MEs_.at("EventTypeCalib"));
    MESet& meEventTypePostCalib(MEs_.at("EventTypePostCalib"));

    if (!l1A_) {
      // majority vote on L1A.. is there no better implementation?
      map<int, int> l1aCounts;
      for (EcalRawDataCollection::const_iterator dcchItr(_dcchs.begin()); dcchItr != _dcchs.end(); ++dcchItr) {
        l1aCounts[dcchItr->getLV1()]++;
      }
      int maxVote(0);
      for (map<int, int>::iterator l1aItr(l1aCounts.begin()); l1aItr != l1aCounts.end(); ++l1aItr) {
        if (l1aItr->second > maxVote) {
          maxVote = l1aItr->second;
          l1A_ = l1aItr->first;
        }
      }
    }

    for (EcalRawDataCollection::const_iterator dcchItr(_dcchs.begin()); dcchItr != _dcchs.end(); ++dcchItr) {
      int dccId(dcchItr->id());

      int dccL1A(dcchItr->getLV1());
      short dccL1AShort(dccL1A & 0xfff);
      int dccBX(dcchItr->getBX());

      meOrbitDiff.fill(getEcalDQMSetupObjects(), dccId, dcchItr->getOrbit() - orbit_);
      meBXDCCDiff.fill(getEcalDQMSetupObjects(), dccId, dccBX - bx_);
      if (dccBX == -1)
        meBXFEInvalid.fill(getEcalDQMSetupObjects(), dccId, 68.5);

      if (dcchItr->getRunNumber() != int(runNumber_))
        meRunNumber.fill(getEcalDQMSetupObjects(), dccId);
      if (dcchItr->getOrbit() != orbit_)
        meOrbit.fill(getEcalDQMSetupObjects(), dccId);
      if (dcchItr->getBasicTriggerType() != triggerType_)
        meTriggerType.fill(getEcalDQMSetupObjects(), dccId);
      if (dccL1A != l1A_)
        meL1ADCC.fill(getEcalDQMSetupObjects(), dccId);
      if (dccBX != bx_)
        meBXDCC.fill(getEcalDQMSetupObjects(), dccId);

      const vector<short>& feStatus(dcchItr->getFEStatus());
      const vector<short>& feBxs(dcchItr->getFEBxs());
      const vector<short>& feL1s(dcchItr->getFELv1());

      double feDesync(0.);
      double statusError(0.);

      for (unsigned iFE(0); iFE < feStatus.size(); iFE++) {
        if (!ccuExists(dccId, iFE + 1))
          continue;

        short status(feStatus[iFE]);

        if (feBxs[iFE] != -1 && dccBX != -1) {
          meBXFEDiff.fill(getEcalDQMSetupObjects(), dccId, feBxs[iFE] - dccBX);
        }
        if (feBxs[iFE] == -1)
          meBXFEInvalid.fill(getEcalDQMSetupObjects(), dccId, iFE + 0.5);

        if (status != BXDesync && status != L1ABXDesync) {  // BX desync not detected in the DCC
          if (feBxs[iFE] != dccBX && feBxs[iFE] != -1 && dccBX != -1) {
            meBXFE.fill(getEcalDQMSetupObjects(), dccId, iFE + 0.5);
            feDesync += 1.;
          }
        }

        if (status != L1ADesync && status != L1ABXDesync) {
          if (feL1s[iFE] + feL1Offset_ != dccL1AShort && feL1s[iFE] != -1 && dccL1AShort != 0) {
            meL1AFE.fill(getEcalDQMSetupObjects(), dccId, iFE + 0.5);
            feDesync += 1.;
          }
        }

        if (iFE >= 68) {
          // FE Status for MEM boxes (towerIds 69 and 70)
          // Plot contains two bins per dccId. Integer number
          // bins correspond to towerId 69 and half integer
          // number bins correspond to towerId 70.
          if (iFE + 1 == 69)
            meFEStatusMEM.fill(getEcalDQMSetupObjects(), dccId + 0.0, status);
          else if (iFE + 1 == 70)
            meFEStatusMEM.fill(getEcalDQMSetupObjects(), dccId + 0.5, status);
          continue;
        }

        DetId id(GetElectronicsMap()->dccTowerConstituents(dccId, iFE + 1).at(0));
        meFEStatus.fill(getEcalDQMSetupObjects(), id, status);
        // Fill FE Status Error Map with error states only
        if (status != Enabled && status != Suppressed && status != FIFOFull && status != FIFOFullL1ADesync &&
            status != ForcedZS)
          meFEStatusErrMapByLumi.fill(getEcalDQMSetupObjects(), id, status);

        switch (status) {
          case Timeout:
          case HeaderError:
          case ChannelId:
          case LinkError:
          case BlockSize:
          case L1ADesync:
          case BXDesync:
          case L1ABXDesync:
          case HParity:
          case VParity:
            statusError += 1.;
            break;
          default:
            continue;
        }
      }

      if (feDesync > 0.) {
        meDesyncByLumi.fill(getEcalDQMSetupObjects(), dccId, feDesync);
        meDesyncTotal.fill(getEcalDQMSetupObjects(), dccId, feDesync);
        meTrendNSyncErrors.fill(getEcalDQMSetupObjects(), double(timestamp_.iLumi), feDesync);
      }
      if (statusError > 0.)
        meFEByLumi.fill(getEcalDQMSetupObjects(), dccId, statusError);

      const vector<short>& tccBx(dcchItr->getTCCBx());
      const vector<short>& tccL1(dcchItr->getTCCLv1());

      if (tccBx.size() == 4) {  // EB uses tccBx[0]; EE uses all
        if (dccId <= kEEmHigh + 1 || dccId >= kEEpLow + 1) {
          for (int iTCC(0); iTCC < 4; iTCC++) {
            if (tccBx[iTCC] != dccBX && tccBx[iTCC] != -1 && dccBX != -1)
              meBXTCC.fill(getEcalDQMSetupObjects(), dccId);

            if (tccL1[iTCC] != dccL1AShort && tccL1[iTCC] != -1 && dccL1AShort != 0)
              meL1ATCC.fill(getEcalDQMSetupObjects(), dccId);
          }
        } else {
          if (tccBx[0] != dccBX && tccBx[0] != -1 && dccBX != -1)
            meBXTCC.fill(getEcalDQMSetupObjects(), dccId);

          if (tccL1[0] != dccL1AShort && tccL1[0] != -1 && dccL1AShort != 0)
            meL1ATCC.fill(getEcalDQMSetupObjects(), dccId);
        }
      }

      short srpBx(dcchItr->getSRPBx());
      short srpL1(dcchItr->getSRPLv1());

      if (srpBx != dccBX && srpBx != -1 && dccBX != -1)
        meBXSRP.fill(getEcalDQMSetupObjects(), dccId);

      if (srpL1 != dccL1AShort && srpL1 != -1 && dccL1AShort != 0)
        meL1ASRP.fill(getEcalDQMSetupObjects(), dccId);

      const int calibBX(3490);

      short runType(dcchItr->getRunType() + 1);
      if (runType < 0 || runType > 22)
        runType = 0;
      if (dccBX < calibBX)
        meEventTypePreCalib.fill(getEcalDQMSetupObjects(), dccId, runType, 1. / 54.);
      else if (dccBX == calibBX)
        meEventTypeCalib.fill(getEcalDQMSetupObjects(), dccId, runType, 1. / 54.);
      else
        meEventTypePostCalib.fill(getEcalDQMSetupObjects(), dccId, runType, 1. / 54.);
    }
  }

  DEFINE_ECALDQM_WORKER(RawDataTask);
}  // namespace ecaldqm
