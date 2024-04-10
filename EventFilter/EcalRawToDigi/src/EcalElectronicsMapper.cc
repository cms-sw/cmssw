#include <cassert>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <EventFilter/EcalRawToDigi/interface/EcalElectronicsMapper.h>
#include <Geometry/EcalMapping/interface/EcalElectronicsMapping.h>
#include <DataFormats/EcalDigi/interface/EBSrFlag.h>
#include <DataFormats/EcalDigi/interface/EESrFlag.h>
#include <EventFilter/EcalRawToDigi/interface/DCCDataUnpacker.h>

EcalElectronicsMapper::EcalElectronicsMapper(unsigned int numbXtalTSamples, unsigned int numbTriggerTSamples)
    : pathToMapFile_(""),
      numbXtalTSamples_(numbXtalTSamples),
      numbTriggerTSamples_(numbTriggerTSamples),
      mappingBuilder_(nullptr) {
  resetPointers();
  setupGhostMap();
}

void EcalElectronicsMapper::resetPointers() {
  // Reset Arrays
  for (unsigned int sm = 0; sm < NUMB_SM; sm++) {
    for (unsigned int fe = 0; fe < NUMB_FE; fe++) {
      for (unsigned int strip = 0; strip < NUMB_STRIP; strip++) {
        for (unsigned int xtal = 0; xtal < NUMB_XTAL; xtal++) {
          //Reset DFrames and xtalDetIds
          xtalDetIds_[sm][fe][strip][xtal] = nullptr;
        }
      }

      //Reset SC Det Ids
      //scDetIds_[sm][fe]=0;
      scEleIds_[sm][fe] = nullptr;
      //srFlags_[sm][fe]=0;
    }
  }

  //Reset TT det Ids
  for (unsigned int tccid = 0; tccid < NUMB_TCC; tccid++) {
    for (unsigned int tpg = 0; tpg < NUMB_FE; tpg++) {
      ttDetIds_[tccid][tpg] = nullptr;
      ttTPIds_[tccid][tpg] = nullptr;
      ttEleIds_[tccid][tpg] = nullptr;
    }
  }

  // reset trigger electronics Id
  for (int tccid = 0; tccid < NUMB_TCC; tccid++) {
    for (int ttid = 0; ttid < TCC_EB_NUMBTTS; ttid++) {
      for (int ps = 0; ps < NUMB_STRIP; ps++) {
        psInput_[tccid][ttid][ps] = nullptr;
      }
    }
  }

  // initialize TCC maps
  for (int tccId = 0; tccId < EcalTriggerElectronicsId::MAX_TCCID; tccId++) {
    for (int psCounter = 0; psCounter < EcalTrigTowerDetId::kEBTowersPerSM * 5; psCounter++) {
      for (int u = 0; u < 2; u++) {
        tTandPs_[tccId][psCounter][u] = -1;
      }
    }
  }

  //Fill map sm id to tcc ids
  std::vector<unsigned int>* ids;
  ids = new std::vector<unsigned int>;
  ids->push_back(1);
  ids->push_back(18);
  ids->push_back(19);
  ids->push_back(36);
  mapSmIdToTccIds_[1] = ids;

  ids = new std::vector<unsigned int>;
  ids->push_back(2);
  ids->push_back(3);
  ids->push_back(20);
  ids->push_back(21);
  mapSmIdToTccIds_[2] = ids;

  ids = new std::vector<unsigned int>;
  ids->push_back(4);
  ids->push_back(5);
  ids->push_back(22);
  ids->push_back(23);
  mapSmIdToTccIds_[3] = ids;

  ids = new std::vector<unsigned int>;
  ids->push_back(6);
  ids->push_back(7);
  ids->push_back(24);
  ids->push_back(25);
  mapSmIdToTccIds_[4] = ids;

  ids = new std::vector<unsigned int>;
  ids->push_back(8);
  ids->push_back(9);
  ids->push_back(26);
  ids->push_back(27);
  mapSmIdToTccIds_[5] = ids;

  ids = new std::vector<unsigned int>;
  ids->push_back(10);
  ids->push_back(11);
  ids->push_back(28);
  ids->push_back(29);
  mapSmIdToTccIds_[6] = ids;

  ids = new std::vector<unsigned int>;
  ids->push_back(12);
  ids->push_back(13);
  ids->push_back(30);
  ids->push_back(31);
  mapSmIdToTccIds_[7] = ids;

  ids = new std::vector<unsigned int>;
  ids->push_back(14);
  ids->push_back(15);
  ids->push_back(32);
  ids->push_back(33);
  mapSmIdToTccIds_[8] = ids;

  ids = new std::vector<unsigned int>;
  ids->push_back(16);
  ids->push_back(17);
  ids->push_back(34);
  ids->push_back(35);
  mapSmIdToTccIds_[9] = ids;

  ids = new std::vector<unsigned int>;
  ids->push_back(73);
  ids->push_back(90);
  ids->push_back(91);
  ids->push_back(108);
  mapSmIdToTccIds_[46] = ids;

  ids = new std::vector<unsigned int>;
  ids->push_back(74);
  ids->push_back(75);
  ids->push_back(92);
  ids->push_back(93);
  mapSmIdToTccIds_[47] = ids;

  ids = new std::vector<unsigned int>;
  ids->push_back(76);
  ids->push_back(77);
  ids->push_back(94);
  ids->push_back(95);
  mapSmIdToTccIds_[48] = ids;

  ids = new std::vector<unsigned int>;
  ids->push_back(78);
  ids->push_back(79);
  ids->push_back(96);
  ids->push_back(97);
  mapSmIdToTccIds_[49] = ids;

  ids = new std::vector<unsigned int>;
  ids->push_back(80);
  ids->push_back(81);
  ids->push_back(98);
  ids->push_back(99);
  mapSmIdToTccIds_[50] = ids;

  ids = new std::vector<unsigned int>;
  ids->push_back(82);
  ids->push_back(83);
  ids->push_back(100);
  ids->push_back(101);
  mapSmIdToTccIds_[51] = ids;

  ids = new std::vector<unsigned int>;
  ids->push_back(84);
  ids->push_back(85);
  ids->push_back(102);
  ids->push_back(103);
  mapSmIdToTccIds_[52] = ids;

  ids = new std::vector<unsigned int>;
  ids->push_back(86);
  ids->push_back(87);
  ids->push_back(104);
  ids->push_back(105);
  mapSmIdToTccIds_[53] = ids;

  ids = new std::vector<unsigned int>;
  ids->push_back(88);
  ids->push_back(89);
  ids->push_back(106);
  ids->push_back(107);
  mapSmIdToTccIds_[54] = ids;

  //Compute data block sizes
  unfilteredFEBlockLength_ = computeUnfilteredFEBlockLength();
  ebTccBlockLength_ = computeEBTCCBlockLength();
  eeTccBlockLength_ = computeEETCCBlockLength();
}

EcalElectronicsMapper::~EcalElectronicsMapper() { deletePointers(); }

void EcalElectronicsMapper::deletePointers() {
  //DETETE ARRAYS
  for (unsigned int sm = 0; sm < NUMB_SM; sm++) {
    for (unsigned int fe = 0; fe < NUMB_FE; fe++) {
      for (unsigned int strip = 0; strip < NUMB_STRIP; strip++) {
        for (unsigned int xtal = 0; xtal < NUMB_XTAL; xtal++)
          delete xtalDetIds_[sm][fe][strip][xtal];
      }

      // if(scDetIds_[sm][fe]){
      //  delete scDetIds_[sm][fe];
      //  delete scEleIds_[sm][fe];
      for (size_t i = 0; i < srFlags_[sm][fe].size(); ++i)
        delete srFlags_[sm][fe][i];
      srFlags_[sm][fe].clear();

      delete scEleIds_[sm][fe];
    }
  }

  // delete trigger electronics Id
  for (int tccid = 0; tccid < NUMB_TCC; tccid++) {
    for (int ttid = 0; ttid < TCC_EB_NUMBTTS; ttid++) {
      for (int ps = 0; ps < NUMB_STRIP; ps++) {
        delete psInput_[tccid][ttid][ps];
      }
    }
  }

  for (unsigned int tccid = 0; tccid < NUMB_TCC; tccid++) {
    for (unsigned int tpg = 0; tpg < NUMB_FE; tpg++) {
      if (ttDetIds_[tccid][tpg]) {
        delete ttDetIds_[tccid][tpg];
        delete ttTPIds_[tccid][tpg];
        delete ttEleIds_[tccid][tpg];
      }
    }
  }

  pathToMapFile_.clear();

  std::map<unsigned int, std::vector<unsigned int>*>::iterator it;
  for (it = mapSmIdToTccIds_.begin(); it != mapSmIdToTccIds_.end(); it++) {
    delete (*it).second;
  }

  mapSmIdToTccIds_.clear();
}

void EcalElectronicsMapper::setEcalElectronicsMapping(const EcalElectronicsMapping* m) {
  mappingBuilder_ = m;
  fillMaps();
}

bool EcalElectronicsMapper::setActiveDCC(unsigned int dccId) {
  bool ret(true);

  //Update active dcc and associated smId
  dccId_ = dccId;

  smId_ = getSMId(dccId_);

  if (!smId_)
    ret = false;

  return ret;
}

bool EcalElectronicsMapper::setDCCMapFilePath(std::string aPath_) {
  //try to open a dccMapFile in the given path
  std::ifstream dccMapFile_(aPath_.c_str());

  //if not successful return false
  if (!dccMapFile_.is_open())
    return false;

  //else close file and accept given path
  dccMapFile_.close();
  pathToMapFile_ = aPath_;

  return true;
}

// bool EcalElectronicsMapper::readDCCMapFile(){

//   //try to open a dccMapFile in the given path
//   std::ifstream dccMapFile_(pathToMapFile_.c_str());

//   //if not successful return false
//   if(!dccMapFile_.is_open()) return false;

//   char lineBuf_[100];
//   unsigned int SMId_,DCCId_;
//   // loop while extraction from file is possible
//   dccMapFile_.getline(lineBuf_,10);       //read line from file
//   while (dccMapFile_.good()) {
//     sscanf(lineBuf_,"%u:%u",&SMId_,&DCCId_);
//     myDCCMap_[SMId_] = DCCId_;
//     dccMapFile_.getline(lineBuf_,10);       //read line from file
//   }

//   return true;

// }

// bool EcalElectronicsMapper::readDCCMapFile(std::string aPath_){
//   //test if path is good
//   edm::FileInPath eff(aPath_);

//   if(!setDCCMapFilePath(eff.fullPath())) return false;

//   //read DCC map file
//   readDCCMapFile();
//   return true;
// }

bool EcalElectronicsMapper::makeMapFromVectors(std::vector<int>& orderedFedUnpackList,
                                               std::vector<int>& orderedDCCIdList) {
  // in case as non standard set of DCCId:FedId pairs was provided
  if (orderedFedUnpackList.size() == orderedDCCIdList.size() && !orderedFedUnpackList.empty()) {
    edm::LogInfo("EcalElectronicsMapper") << "DCCIdList/FedUnpackList lists given. Being loaded.";

    std::string correspondence("list of pairs DCCId:FedId :  ");
    char onePair[50];
    for (int v = 0; v < ((int)orderedFedUnpackList.size()); v++) {
      myDCCMap_[orderedDCCIdList[v]] = orderedFedUnpackList[v];

      sprintf(onePair, "  %d:%d", orderedDCCIdList[v], orderedFedUnpackList[v]);
      std::string tmp(onePair);
      correspondence += tmp;
    }
    edm::LogInfo("EcalElectronicsMapper") << correspondence;

  } else {  // default set of DCCId:FedId for ECAL

    edm::LogInfo("EcalElectronicsMapper") << "No input DCCIdList/FedUnpackList lists given for ECAL unpacker"
                                          << "(or given with different number of elements). "
                                          << " Loading default association DCCIdList:FedUnpackList,"
                                          << "i.e.  1:601 ... 53:653,  54:654.";

    for (unsigned int v = 1; v <= 54; v++) {
      myDCCMap_[v] = (v + 600);
    }
  }

  return true;
}

std::ostream& operator<<(std::ostream& o, const EcalElectronicsMapper& aMapper_) {
  //print class information
  o << "---------------------------------------------------------";

  if (aMapper_.pathToMapFile_.empty()) {
    o << "No correct input for DCC map has been given yet...";
  } else {
    o << "DCC Map (Map file: " << aMapper_.pathToMapFile_ << " )"
      << "SM id\t\tDCCid ";

    //get DCC map and iterator
    std::map<unsigned int, unsigned int> aMap;
    aMap = aMapper_.myDCCMap_;
    std::map<unsigned int, unsigned int>::iterator iter;

    //print info contained in map
    for (iter = aMap.begin(); iter != aMap.end(); iter++)
      o << iter->first << "\t\t" << iter->second;
  }

  o << "---------------------------------------------------------";
  return o;
}

unsigned int EcalElectronicsMapper::computeUnfilteredFEBlockLength() {
  return ((numbXtalTSamples_ - 2) / 4 + 1) * 25 + 1;
}

unsigned int EcalElectronicsMapper::computeEBTCCBlockLength() {
  unsigned int nTT = 68;
  unsigned int tf;

  //TCC block size: header (8 bytes) + 17 words with 4 trigger primitives (17*8bytes)
  if ((nTT * numbTriggerTSamples_) < 4 || (nTT * numbTriggerTSamples_) % 4)
    tf = 1;
  else
    tf = 0;

  return 1 + ((nTT * numbTriggerTSamples_) / 4) + tf;
}

unsigned int EcalElectronicsMapper::computeEETCCBlockLength() {
  //Todo : implement multiple tt samples for the endcap
  return 9;
}

bool EcalElectronicsMapper::isTCCExternal(unsigned int TCCId) {
  if ((NUMB_TCC_EE_MIN_EXT_MIN <= TCCId && TCCId <= NUMB_TCC_EE_MIN_EXT_MAX) ||
      (NUMB_TCC_EE_PLU_EXT_MIN <= TCCId && TCCId <= NUMB_TCC_EE_PLU_EXT_MAX))
    return true;
  else
    return false;
}

unsigned int EcalElectronicsMapper::getDCCId(unsigned int aSMId_) const {
  //get iterator for SM id
  std::map<unsigned int, unsigned int>::const_iterator it = myDCCMap_.find(aSMId_);

  //check if SMid exists and return DCC id
  if (it != myDCCMap_.end())
    return it->second;

  //error return
  if (!DCCDataUnpacker::silentMode_) {
    edm::LogError("IncorrectMapping") << "DCC requested for SM id: " << aSMId_ << " not found";
  }
  return 0;
}

unsigned int EcalElectronicsMapper::getSMId(unsigned int aDCCId_) const {
  //get iterator map
  std::map<unsigned int, unsigned int>::const_iterator it;

  //try to find SM id for given DCC id
  for (it = myDCCMap_.begin(); it != myDCCMap_.end(); it++)
    if (it->second == aDCCId_)
      return it->first;

  //error return
  if (!DCCDataUnpacker::silentMode_) {
    edm::LogError("IncorrectMapping") << "SM requested DCC id: " << aDCCId_ << " not found";
  }
  return 0;
}

void EcalElectronicsMapper::fillMaps() {
  for (int smId = 1; smId <= 54; smId++) {
    // Fill EB arrays
    if (smId > 9 && smId < 46) {
      for (int feChannel = 1; feChannel <= 68; feChannel++) {
        unsigned int tccId = smId + TCCID_SMID_SHIFT_EB;

        // Builds Ecal Trigger Tower Det Id

        unsigned int rawid = (mappingBuilder_->getTrigTowerDetId(tccId, feChannel)).rawId();
        EcalTrigTowerDetId* ttDetId = new EcalTrigTowerDetId(rawid);
        ttDetIds_[tccId - 1][feChannel - 1] = ttDetId;
        EcalElectronicsId* ttEleId = new EcalElectronicsId(smId, feChannel, 1, 1);
        ttEleIds_[tccId - 1][feChannel - 1] = ttEleId;
        EcalTriggerPrimitiveDigi* tp = new EcalTriggerPrimitiveDigi(*ttDetId);
        tp->setSize(numbTriggerTSamples_);
        for (unsigned int i = 0; i < numbTriggerTSamples_; i++) {
          tp->setSample(i, EcalTriggerPrimitiveSample(0));
        }
        ttTPIds_[tccId - 1][feChannel - 1] = tp;

        // build pseudostrip input data digi
        for (int ps = 1; ps <= 5; ps++) {
          psInput_[tccId - 1][feChannel - 1][ps - 1] =
              new EcalPseudoStripInputDigi(EcalTriggerElectronicsId(tccId, feChannel, ps, 1));
          psInput_[tccId - 1][feChannel - 1][ps - 1]->setSize(1);
          psInput_[tccId - 1][feChannel - 1][ps - 1]->setSample(0, EcalPseudoStripInputSample(0));
        }

        // Buil SRP Flag
        srFlags_[smId - 1][feChannel - 1].push_back(new EBSrFlag(*ttDetId, 0));

        //only one element for barrel: 1-to-1 correspondance between
        //DCC channels and EB trigger tower:
        assert(srFlags_[smId - 1][feChannel - 1].size() == 1);

        for (unsigned int stripId = 1; stripId <= 5; stripId++) {
          for (unsigned int xtalId = 1; xtalId <= 5; xtalId++) {
            EcalElectronicsId eid(smId, feChannel, stripId, xtalId);
            EBDetId* detId = new EBDetId((mappingBuilder_->getDetId(eid)).rawId());
            xtalDetIds_[smId - 1][feChannel - 1][stripId - 1][xtalId - 1] = detId;

          }  // close loop over xtals
        }    // close loop over strips

      }  // close loop over fechannels

    }  //close loop over sm ids in the EB
    // Fill EE arrays (Todo : waiting SC correction)

    else {
      std::vector<unsigned int>* pTCCIds = mapSmIdToTccIds_[smId];
      std::vector<unsigned int>::iterator it;

      for (it = pTCCIds->begin(); it != pTCCIds->end(); it++) {
        unsigned int tccId = *it;

        // creating arrays of pointers for trigger objects
        for (unsigned int towerInTCC = 1; towerInTCC <= numChannelsInDcc_[smId - 1]; towerInTCC++) {
          // Builds Ecal Trigger Tower Det Id
          EcalTrigTowerDetId ttDetId = mappingBuilder_->getTrigTowerDetId(tccId, towerInTCC);

          ttDetIds_[tccId - 1][towerInTCC - 1] = new EcalTrigTowerDetId(ttDetId.rawId());
          EcalTriggerPrimitiveDigi* tp = new EcalTriggerPrimitiveDigi(ttDetId);
          tp->setSize(numbTriggerTSamples_);
          for (unsigned int i = 0; i < numbTriggerTSamples_; i++) {
            tp->setSample(i, EcalTriggerPrimitiveSample(0));
          }

          ttTPIds_[tccId - 1][towerInTCC - 1] = tp;

          // build pseudostrip input data digi
          for (int ps = 1; ps <= 5; ps++) {
            psInput_[tccId - 1][towerInTCC - 1][ps - 1] =
                new EcalPseudoStripInputDigi(EcalTriggerElectronicsId(tccId, towerInTCC, ps, 1));
            psInput_[tccId - 1][towerInTCC - 1][ps - 1]->setSize(1);
            psInput_[tccId - 1][towerInTCC - 1][ps - 1]->setSample(0, EcalPseudoStripInputSample(0));
          }
        }
      }

      // creating arrays of pointers for digi objects
      for (unsigned int feChannel = 1; feChannel <= numChannelsInDcc_[smId - 1]; feChannel++) {
        // to avoid gap in CCU_id's
        if ((smId == SECTOR_EEM_CCU_JUMP || smId == SECTOR_EEP_CCU_JUMP) &&
            (MIN_CCUID_JUMP <= feChannel && feChannel <= MAX_CCUID_JUMP))
          continue;

        std::vector<EcalScDetId> scDetIds = mappingBuilder_->getEcalScDetId(smId, feChannel);
        // scDetIds_[smId-1][feChannel-1] = new EcalScDetId(scDetId.rawId());
        scEleIds_[smId - 1][feChannel - 1] = new EcalElectronicsId(smId, feChannel, 1, 1);

        for (size_t i = 0; i < scDetIds.size(); ++i) {
          // std::cout << __FILE__ << ":" << __LINE__ << ": "
          //            << "(DCC,RU) = (" <<  smId << "," << feChannel
          //            << ") -> " << scDetIds[i] << "\n";

          srFlags_[smId - 1][feChannel - 1].push_back(new EESrFlag(EcalScDetId(scDetIds[i].rawId()), 0));
        }
        //usually only one element 1 DCC channel <-> 1 SC
        //in few case two or three elements: partial SCs grouped.
        assert(srFlags_[smId - 1][feChannel - 1].size() <= 3);

        std::vector<DetId> ecalDetIds = mappingBuilder_->dccTowerConstituents(smId, feChannel);
        std::vector<DetId>::iterator it;

        //EEDetIds
        for (it = ecalDetIds.begin(); it != ecalDetIds.end(); it++) {
          EcalElectronicsId ids = mappingBuilder_->getElectronicsId((*it));

          int stripId = ids.stripId();
          int xtalId = ids.xtalId();

          EEDetId* detId = new EEDetId((*it).rawId());
          xtalDetIds_[smId - 1][feChannel - 1][stripId - 1][xtalId - 1] = detId;
        }  // close loop over tower constituents

      }  // close loop over  FE Channels

    }  // closing loop over sm ids in EE
  }

  // developing mapping for pseudostrip input data: (tccId,psNumber)->(tccId,towerId,psId)
  // initializing array for pseudostrip data
  short numStripInTT[EcalTriggerElectronicsId::MAX_TCCID][EcalTrigTowerDetId::kEBTowersPerSM];
  for (int tccId = 0; tccId < EcalTriggerElectronicsId::MAX_TCCID; tccId++) {
    for (int tt = 0; tt < EcalTrigTowerDetId::kEBTowersPerSM; tt++) {
      numStripInTT[tccId][tt] = -2;
    }
  }

  // assumption: if ps_max is the largest pseudostripId within a trigger tower
  // all the pseudostrip 1 ...  ps_max are actually present
  std::vector<DetId>::iterator theTCCConstituent;
  for (int tccId = 0; tccId < EcalTriggerElectronicsId::MAX_TCCID; tccId++) {
    // loop over all constituents of a TCC and collect
    // the largest pseudostripId within each trigger tower
    std::vector<DetId> tccConstituents = mappingBuilder_->tccConstituents(tccId + 1);

    for (theTCCConstituent = tccConstituents.begin(); theTCCConstituent != tccConstituents.end(); theTCCConstituent++) {
      int towerId = (mappingBuilder_->getTriggerElectronicsId(*theTCCConstituent)).ttId();
      int ps = (mappingBuilder_->getTriggerElectronicsId(*theTCCConstituent)).pseudoStripId();
      if (ps > numStripInTT[tccId][towerId - 1])
        numStripInTT[tccId][towerId - 1] = ps;

      //std::cout << "tccId: " << (tccId+1) << "    towerId: " << towerId
      // << "    ps: " << ps << "    numStripInTT: " << numStripInTT[tccId][towerId-1] << std::endl;

    }  // loop on TCC constituents

  }  // loop on TCC's

  int psCounter;
  for (int tccId = 0; tccId < EcalTriggerElectronicsId::MAX_TCCID; tccId++) {
    // resetting pseudostrip counter at each new TCC
    psCounter = 0;
    for (int towerId = 0; towerId < EcalTrigTowerDetId::kEBTowersPerSM; towerId++) {
      // if there's not a given towerId, numStripInTT==-1
      for (int ps = 0; ps < numStripInTT[tccId][towerId]; ps++) {
        tTandPs_[tccId][psCounter][0] = towerId + 1;
        tTandPs_[tccId][psCounter][1] = ps + 1;
        psCounter++;
      }  // loop on TCC's
    }    // loop on towers in TCC
  }      // loop in ps in tower

  //   for (int tccId=0; tccId<EcalTriggerElectronicsId::MAX_TCCID; tccId++) {
  //       for (int psCounter=0; psCounter<EcalTrigTowerDetId::kEBTowersPerSM*5; psCounter++) {
  //           std::cout << "tccId: " << (tccId+1) << "    counter: " << (psCounter+1)
  //                     << " tt: " << tTandPs_[tccId][psCounter][0]
  //                     << " ps: " << tTandPs_[tccId][psCounter][1]
  //                     << std::endl;
  //       } }

  // Note:
  // for TCC 48 in EE, pseudostrip data in the interval:
  // + 1-30 is good pseudostrip data
  // + 31-42 is a duplication of the last 12 ps of the previous block, and needs be ignored
  // + 43-60 is further good pseudostrip data

  for (int tcc = EcalTriggerElectronicsId::MIN_TCCID_EEM; tcc <= EcalTriggerElectronicsId::MAX_TCCID_EEM; tcc++) {
    int tccId = tcc - 1;
    short tTandPs_tmp[18][2];

    // store entries _after_ the pseudostrip data which gets duplicated
    for (int psCounter = 30; psCounter < 48; psCounter++) {
      tTandPs_tmp[psCounter - 30][0] = tTandPs_[tccId][psCounter][0];
      tTandPs_tmp[psCounter - 30][1] = tTandPs_[tccId][psCounter][1];
    }

    // duplication
    for (int psCounter = 18; psCounter < 30; psCounter++) {
      tTandPs_[tccId][psCounter + 12][0] = tTandPs_[tccId][psCounter][0];
      tTandPs_[tccId][psCounter + 12][1] = tTandPs_[tccId][psCounter][1];
    }

    // append stoed
    for (int psCounter = 42; psCounter < 60; psCounter++) {
      tTandPs_[tccId][psCounter][0] = tTandPs_tmp[psCounter - 42][0];
      tTandPs_[tccId][psCounter][1] = tTandPs_tmp[psCounter - 42][1];
    }

  }  // loop on EEM TCC's

  for (int tcc = EcalTriggerElectronicsId::MIN_TCCID_EEP; tcc <= EcalTriggerElectronicsId::MAX_TCCID_EEP; tcc++) {
    int tccId = tcc - 1;
    short tTandPs_tmp[18][2];

    // store entries _after_ the pseudostrip data which gets duplicated
    for (int psCounter = 30; psCounter < 48; psCounter++) {
      tTandPs_tmp[psCounter - 30][0] = tTandPs_[tccId][psCounter][0];
      tTandPs_tmp[psCounter - 30][1] = tTandPs_[tccId][psCounter][1];
    }

    // duplication
    for (int psCounter = 18; psCounter < 30; psCounter++) {
      tTandPs_[tccId][psCounter + 12][0] = tTandPs_[tccId][psCounter][0];
      tTandPs_[tccId][psCounter + 12][1] = tTandPs_[tccId][psCounter][1];
    }

    // append stoed
    for (int psCounter = 42; psCounter < 60; psCounter++) {
      tTandPs_[tccId][psCounter][0] = tTandPs_tmp[psCounter - 42][0];
      tTandPs_[tccId][psCounter][1] = tTandPs_tmp[psCounter - 42][1];
    }

  }  // loop on EEP TCC's

  //for (int tccId=0; tccId<EcalTriggerElectronicsId::MAX_TCCID; tccId++) {
  //for (int psCounter=0; psCounter<EcalTrigTowerDetId::kEBTowersPerSM*5; psCounter++) {
  //std::cout << "AF tccId: " << (tccId+1) << "    counter: " << (psCounter+1)
  //<< " tt: " << tTandPs_[tccId][psCounter][0]
  //<< " ps: " << tTandPs_[tccId][psCounter][1]
  //<< std::endl;
  // } }
}

void EcalElectronicsMapper::setupGhostMap() {
  // number of 'ghost' VFEs
  const int n = 44;

  // here is a list of all 'ghost' VFEs
  // in format {FED, CCU, VFE}
  const struct {
    int FED, CCU, VFE;
  } v[n] = {{601, 10, 5}, {601, 34, 3}, {601, 34, 4}, {601, 34, 5}, {602, 32, 5}, {603, 12, 5}, {603, 30, 5},
            {604, 12, 5}, {604, 30, 5}, {605, 32, 5}, {606, 10, 5}, {606, 34, 3}, {606, 34, 4}, {606, 34, 5},
            {608, 27, 3}, {608, 27, 4}, {608, 27, 5}, {608, 3, 3},  {608, 3, 4},  {608, 3, 5},  {608, 30, 5},
            {608, 6, 5},  {646, 10, 5}, {646, 34, 3}, {646, 34, 4}, {646, 34, 5}, {647, 32, 5}, {648, 12, 5},
            {648, 30, 5}, {649, 12, 5}, {649, 30, 5}, {650, 32, 5}, {651, 10, 5}, {651, 34, 3}, {651, 34, 4},
            {651, 34, 5}, {653, 27, 3}, {653, 27, 4}, {653, 27, 5}, {653, 3, 3},  {653, 3, 4},  {653, 3, 5},
            {653, 30, 5}, {653, 6, 5}};

  for (int i = 0; i < n; ++i)
    ghost_[v[i].FED][v[i].CCU][v[i].VFE] = true;
}

bool EcalElectronicsMapper::isGhost(const int FED, const int CCU, const int VFE) {
  if (ghost_.find(FED) == ghost_.end())
    return false;

  if (ghost_[FED].find(CCU) == ghost_[FED].end())
    return false;

  if (ghost_[FED][CCU].find(VFE) == ghost_[FED][CCU].end())
    return false;

  return true;
}

// number of readout channels (TT in EB, SC in EE) in a DCC
const unsigned int EcalElectronicsMapper::numChannelsInDcc_[NUMB_SM] = {
    34, 32, 33, 33, 32, 34, 33, 41, 33,                                      // EE -
    68, 68, 68, 68, 68, 68, 68, 68, 68, 68,                                  // EB-
    68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,  // EB+
    68, 68, 68, 68, 68, 68, 68, 68, 34, 32, 33, 33, 32, 34, 33, 41, 33};     // EE+
