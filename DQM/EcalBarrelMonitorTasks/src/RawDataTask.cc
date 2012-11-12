#include "../interface/RawDataTask.h"

#include "FWCore/Framework/interface/Event.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/FEFlags.h"

#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"

namespace ecaldqm {

  RawDataTask::RawDataTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "RawDataTask"),
    hltTaskMode_(_commonParams.getUntrackedParameter<int>("hltTaskMode")),
    l1A_(0),
    orbit_(0),
    bx_(0),
    triggerType_(0),
    feL1Offset_(0)
  {
    collectionMask_[kLumiSection] = true;
    collectionMask_[kSource] = true;
    collectionMask_[kEcalRawData] = true;
  }

  void
  RawDataTask::setDependencies(DependencySet& _dependencies)
  {
    _dependencies.push_back(Dependency(kEcalRawData, kSource));
  }

  void
  RawDataTask::bookMEs()
  {
    if(hltTaskMode_ != 1){
      std::string eventTypes[nEventTypes];
      eventTypes[0] = "UNKNOWN";
      // From DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h; should move to reflex
      eventTypes[EcalDCCHeaderBlock::COSMIC + 1] = "COSMIC";
      eventTypes[EcalDCCHeaderBlock::BEAMH4 + 1] = "BEAMH4";
      eventTypes[EcalDCCHeaderBlock::BEAMH2 + 1] = "BEAMH2";
      eventTypes[EcalDCCHeaderBlock::MTCC + 1] = "MTCC";
      eventTypes[EcalDCCHeaderBlock::LASER_STD + 1] = "LASER_STD";
      eventTypes[EcalDCCHeaderBlock::LASER_POWER_SCAN + 1] = "LASER_POWER_SCAN";
      eventTypes[EcalDCCHeaderBlock::LASER_DELAY_SCAN + 1] = "LASER_DELAY_SCAN";
      eventTypes[EcalDCCHeaderBlock::TESTPULSE_SCAN_MEM + 1] = "TESTPULSE_SCAN_MEM";
      eventTypes[EcalDCCHeaderBlock::TESTPULSE_MGPA + 1] = "TESTPULSE_MGPA";
      eventTypes[EcalDCCHeaderBlock::PEDESTAL_STD + 1] = "PEDESTAL_STD";
      eventTypes[EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN + 1] = "PEDESTAL_OFFSET_SCAN";
      eventTypes[EcalDCCHeaderBlock::PEDESTAL_25NS_SCAN + 1] = "PEDESTAL_25NS_SCAN";
      eventTypes[EcalDCCHeaderBlock::LED_STD + 1] = "LED_STD";
      eventTypes[EcalDCCHeaderBlock::PHYSICS_GLOBAL + 1] = "PHYSICS_GLOBAL";
      eventTypes[EcalDCCHeaderBlock::COSMICS_GLOBAL + 1] = "COSMICS_GLOBAL";
      eventTypes[EcalDCCHeaderBlock::HALO_GLOBAL + 1] = "HALO_GLOBAL";
      eventTypes[EcalDCCHeaderBlock::LASER_GAP + 1] = "LASER_GAP";
      eventTypes[EcalDCCHeaderBlock::TESTPULSE_GAP + 1] = "TESTPULSE_GAP";
      eventTypes[EcalDCCHeaderBlock::PEDESTAL_GAP + 1] = "PEDESTAL_GAP";
      eventTypes[EcalDCCHeaderBlock::LED_GAP + 1] = "LED_GAP";
      eventTypes[EcalDCCHeaderBlock::PHYSICS_LOCAL + 1] = "PHYSICS_LOCAL";
      eventTypes[EcalDCCHeaderBlock::COSMICS_LOCAL + 1] = "COSMICS_LOCAL";
      eventTypes[EcalDCCHeaderBlock::HALO_LOCAL + 1] = "HALO_LOCAL";
      eventTypes[EcalDCCHeaderBlock::CALIB_LOCAL + 1] = "CALIB_LOCAL";

      std::string statuses[nFEFlags];
      statuses[Enabled] = "ENABLED";
      statuses[Disabled] = "DISABLED";
      statuses[Timeout] = "TIMEOUT";
      statuses[HeaderError] = "HEADERERROR";
      statuses[ChannelId] = "CHANNELID";
      statuses[LinkError] = "LINKERROR";
      statuses[BlockSize] = "BLOCKSIZE";
      statuses[Suppressed] = "SUPPRESSED";
      statuses[FIFOFull] = "FIFOFULL";
      statuses[L1ADesync] = "L1ADESYNC";
      statuses[BXDesync] = "BXDESYNC";
      statuses[L1ABXDesync] = "L1ABXDESYNC";
      statuses[FIFOFullL1ADesync] = "FIFOFULLL1ADESYNC";
      statuses[HParity] = "HPARITY";
      statuses[VParity] = "VPARITY";
      statuses[ForcedZS] = "FORCEDZS";

      for(MESetCollection::iterator mItr(MEs_.begin()); mItr != MEs_.end(); ++mItr){
        if(mItr->first == "FEDEntries" || mItr->first == "FEDFatal") continue;
	mItr->second->book();
      }
      MEs_["FEByLumi"]->setLumiFlag();

      for(int i(1); i <= nEventTypes; i++){
	MEs_["EventTypePreCalib"]->setBinLabel(-1, i, eventTypes[i - 1], 1);
	MEs_["EventTypeCalib"]->setBinLabel(-1, i, eventTypes[i - 1], 1);
	MEs_["EventTypePostCalib"]->setBinLabel(-1, i, eventTypes[i - 1], 1);
      }

      for(int i(1); i <= nFEFlags; i++)
	MEs_["FEStatus"]->setBinLabel(-1, i, statuses[i - 1], 2);
    }

    if(hltTaskMode_ != 0){
      MEs_["FEDEntries"]->book();
      MEs_["FEDFatal"]->book();
    }
  }

  void
  RawDataTask::beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &)
  {
    MEs_["DesyncByLumi"]->reset();
    MEs_["FEByLumi"]->reset();
  }

  void
  RawDataTask::beginEvent(const edm::Event &_evt, const edm::EventSetup &)
  {
    orbit_ = _evt.orbitNumber() & 0xffffff;
    bx_ = _evt.bunchCrossing() & 0xfff;
    triggerType_ = _evt.experimentType() & 0xf;
    l1A_ = 0;
    feL1Offset_ = _evt.isRealData() ? 1 : 0;
  }

  void
  RawDataTask::runOnSource(const FEDRawDataCollection &_fedRaw, Collections)
  {
    MESet* meFEDEntries(MEs_["FEDEntries"]);
    //    MESet* meFEDFatal(MEs_["FEDFatal"]);
    MESet* meCRC(MEs_["CRC"]);

    // Get GT L1 info
    const FEDRawData &gtFED(_fedRaw.FEDData(812));
    if(gtFED.size() > sizeof(uint64_t)){ // FED header is one 64 bit word
      uint32_t *halfHeader((uint32_t *)gtFED.data());
      l1A_ = *(halfHeader + 1) & 0xffffff;
    }

    for(unsigned iFED(601); iFED <= 654; iFED++){
      const FEDRawData& fedData(_fedRaw.FEDData(iFED));
      unsigned length(fedData.size() / sizeof(uint64_t));
      if(length > 1){ // FED header is one 64 bit word
	meFEDEntries->fill(iFED - 600);

 	const uint64_t* pData(reinterpret_cast<const uint64_t*>(fedData.data()));
 	bool crcError((pData[length - 1] >> 2) & 0x1);

	if(crcError){
          //          meFEDFatal->fill(iFED - 600);
          meCRC->fill(iFED - 600);
        }
      }
    }
  }
  
  void
  RawDataTask::runOnRawData(const EcalRawDataCollection &_dcchs, Collections)
  {
    using namespace std;

    MESet* meRunNumber(MEs_["RunNumber"]);
    MESet* meOrbit(MEs_["Orbit"]);
    MESet* meTriggerType(MEs_["TriggerType"]);
    MESet* meL1ADCC(MEs_["L1ADCC"]);
    MESet* meBXDCC(MEs_["BXDCC"]);
    MESet* meBXFE(MEs_["BXFE"]);
    MESet* meL1AFE(MEs_["L1AFE"]);
    MESet* meFEStatus(MEs_["FEStatus"]);
    MESet* meDesyncByLumi(MEs_["DesyncByLumi"]);
    MESet* meDesyncTotal(MEs_["DesyncTotal"]);
    MESet* meFEByLumi(MEs_["FEByLumi"]);
    MESet* meBXTCC(MEs_["BXTCC"]);
    MESet* meL1ATCC(MEs_["L1ATCC"]);
    MESet* meBXSRP(MEs_["BXSRP"]);
    MESet* meL1ASRP(MEs_["L1ASRP"]);
    MESet* meTrendNSyncErrors(online ? MEs_["L1ATCC"] : 0);
    MESet* meEventTypePreCalib(MEs_["EventTypePreCalib"]);
    MESet* meEventTypeCalib(MEs_["EventTypeCalib"]);
    MESet* meEventTypePostCalib(MEs_["EventTypePostCalib"]);

    if(hltTaskMode_ == 1) return;

    if(!l1A_){
      // majority vote on L1A.. is there no better implementation?
      map<int, int> l1aCounts;
      for(EcalRawDataCollection::const_iterator dcchItr(_dcchs.begin()); dcchItr != _dcchs.end(); ++dcchItr){
	l1aCounts[dcchItr->getLV1()]++;
      }
      int maxVote(0);
      for(map<int, int>::iterator l1aItr(l1aCounts.begin()); l1aItr != l1aCounts.end(); ++l1aItr){
	if(l1aItr->second > maxVote){
	  maxVote = l1aItr->second;
	  l1A_ = l1aItr->first;
	}
      }
    }

    for(EcalRawDataCollection::const_iterator dcchItr(_dcchs.begin()); dcchItr != _dcchs.end(); ++dcchItr){
      unsigned dccId(dcchItr->id());

      int dccL1A(dcchItr->getLV1());
      short dccL1AShort(dccL1A & 0xfff);
      int dccBX(dcchItr->getBX());

      if(dcchItr->getRunNumber() != int(iRun)) meRunNumber->fill(dccId);
      if(dcchItr->getOrbit() != orbit_) meOrbit->fill(dccId);
      if(dcchItr->getBasicTriggerType() != triggerType_) meTriggerType->fill(dccId);
      if(dccL1A != l1A_) meL1ADCC->fill(dccId);
      if(dccBX != bx_) meBXDCC->fill(dccId);

      const vector<short> &feStatus(dcchItr->getFEStatus());
      const vector<short> &feBxs(dcchItr->getFEBxs());
      const vector<short> &feL1s(dcchItr->getFELv1());

      double feDesync(0.);
      double statusError(0.);

      for(unsigned iFE(0); iFE < feStatus.size(); iFE++){
	if(!ccuExists(dccId, iFE + 1)) continue;

	short status(feStatus[iFE]);

	if(status != BXDesync && status != L1ABXDesync){ // BX desync not detected in the DCC
	  if(feBxs[iFE] != dccBX && feBxs[iFE] != -1 && dccBX != -1){
	    meBXFE->fill(dccId, iFE + 0.5);
	    feDesync += 1.;
	  }
	}

	if(status != L1ADesync && status != L1ABXDesync){
	  if(feL1s[iFE] + feL1Offset_ != dccL1AShort && feL1s[iFE] != -1 && dccL1AShort != 0){
	    meL1AFE->fill(dccId, iFE + 0.5);
	    feDesync += 1.;
	  }
	}

	if(iFE >= 68) continue;

	DetId id(getElectronicsMap()->dccTowerConstituents(dccId, iFE + 1).at(0));
	meFEStatus->fill(id, status + 0.5);

	switch(status){
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

      if(feDesync > 0.){
        meDesyncByLumi->fill(dccId, feDesync);
        meDesyncTotal->fill(dccId, feDesync);
        if(online) meTrendNSyncErrors->fill(double(iLumi), feDesync);
      }
      if(statusError > 0.)
        meFEByLumi->fill(dccId, statusError);

      const vector<short> &tccBx(dcchItr->getTCCBx());
      const vector<short> &tccL1(dcchItr->getTCCLv1());

      if(tccBx.size() == 4){ // EB uses tccBx[0]; EE uses all
	if(dccId <= kEEmHigh + 1 || dccId >= kEEpLow + 1){
	  for(int iTCC(0); iTCC < 4; iTCC++){

	    if(tccBx[iTCC] != dccBX && tccBx[iTCC] != -1 && dccBX != -1)
	      meBXTCC->fill(dccId);

	    if(tccL1[iTCC] != dccL1AShort && tccL1[iTCC] != -1 && dccL1AShort != 0)
	      meL1ATCC->fill(dccId);

	  }
	}else{

	  if(tccBx[0] != dccBX && tccBx[0] != -1 && dccBX != -1)
            meBXTCC->fill(dccId);

	  if(tccL1[0] != dccL1AShort && tccL1[0] != -1 && dccL1AShort != 0)
	    meL1ATCC->fill(dccId);

	}
      }

      short srpBx(dcchItr->getSRPBx());
      short srpL1(dcchItr->getSRPLv1());

      if(srpBx != dccBX && srpBx != -1 && dccBX != -1)
	meBXSRP->fill(dccId);

      if(srpL1 != dccL1AShort && srpL1 != -1 && dccL1AShort != 0)
	meL1ASRP->fill(dccId);

      const int calibBX(3490);

      short runType(dcchItr->getRunType() + 1);
      if(runType < 0 || runType > 22) runType = 0;
      if(dccBX < calibBX) meEventTypePreCalib->fill(dccId, runType + 0.5, 1. / 54.);
      else if(dccBX == calibBX) meEventTypeCalib->fill(dccId, runType + 0.5, 1. / 54.);
      else meEventTypePostCalib->fill(dccId, runType + 0.5, 1. / 54.);

    }
  }

  DEFINE_ECALDQM_WORKER(RawDataTask);
}
