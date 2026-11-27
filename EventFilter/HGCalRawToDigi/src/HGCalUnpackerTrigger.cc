#include "EventFilter/HGCalRawToDigi/interface/HGCalUnpackerTrigger.h"
#include "EventFilter/HGCalRawToDigi/interface/TPGFEDataformat.hh"
#include "EventFilter/HGCalRawToDigi/interface/TpgSubpacketHeader.h"
#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"
#include "DataFormats/HGCalDigi/interface/HGCalTriggerDefinitions.h"
#include "DataFormats/HGCalDigi/interface/HGCalFEDTriggerPacketInfoHost.h"
using namespace hgcal;

namespace {
  void convertElinksToTcRawData(hgcal::econt::Type type,
                                unsigned nTc,
                                const uint32_t* v,
                                TPGFEDataformat::TcRawDataPacket& vTcrdp) {
    std::vector<TPGFEDataformat::TcRawData>& vTc(vTcrdp.setTcData());

#ifdef EDM_ML_DEBUG
    for (unsigned i(0); i < 2; i++) {
      LogDebug("[HGCalUnpackerTrigger]") << "Elink " << i << " = 0x" << std::hex << std::setw(8) << std::setfill('0')
                                         << v[i] << std::dec << std::endl;
    }
#endif

    // ECONT header
    unsigned bx(v[0] >> ECONT_FRAME::HEADER_POS);

    LogDebug("[HGCalUnpackerTrigger]") << std::hex << " 32bit word: " << v[0] << std::dec << " ECONT header: " << bx
                                       << std::endl;

    if(nTc <= 0){
      cms::Exception ex("ValueError");
      ex << "nTc "
      << nTc
      << " <= 0";
      ex.addContext("Calling convertElinksToTcRawData()");
      throw ex;
    }
    // assert(nTc > 0);

    bool bitMap(false);
    if (type == hgcal::econt::BestC)
      bitMap = (nTc > 7);

    vTc.resize(0);

    unsigned lastWord(0);
    unsigned lastBit(28);
    uint64_t d(v[0]);

    if (type == hgcal::econt::BestC) {
      lastBit -= 8;
      vTcrdp.setTBM(type, bx, ((d >> lastBit) & 0xff));
      if (false)
        vTcrdp.print();
    } else
      vTcrdp.setTBM(type, bx, 0);

    if (!bitMap) {
      for (unsigned tc(0); tc < nTc; tc++) {
        if (lastBit < 6) {
          d = (d << 32);
          lastWord++;
          //if(v.size()>lastWord) d|=v[lastWord];
          d |= v[lastWord];
          lastBit += 32;
        }

        if (type == hgcal::econt::BestC) {
          lastBit -= 6;
          uint8_t tcAddr = ((d >> lastBit) & ECONT_FRAME::BC_LO_TCADDR_MASK);
          LogDebug("[HGCalUnpackerTrigger]")
              << std::hex << ", d-word : 0x" << std::setfill('0') << std::setw(8) << (d >> lastBit)
              << ", masked-d-word : 0x" << std::setfill('0') << std::setw(8) << uint16_t(tcAddr) << std::dec
              << std::setfill(' ') << std::endl;

          if (tcAddr > ECONT_FRAME::BC_MAX_TCADDR) {
            throw cms::Exception("Stage1IORecoverable")
                << "convertElinksToTcRawData: TC address > " << ECONT_FRAME::BC_MAX_TCADDR << "\n"
                << "  tcAddr        = " << uint16_t(tcAddr) << "\n"
                << "  type       = " << type;
            ;
          }
          vTc.push_back(TPGFEDataformat::TcRawData(type, tcAddr, 0));
        }
        if (type == hgcal::econt::STC4A) {
          lastBit -= 2;
          vTc.push_back(TPGFEDataformat::TcRawData(type, ((d >> lastBit) & 0x03), 0));
        }
        if (type == hgcal::econt::STC4B) {
          lastBit -= 2;
          vTc.push_back(TPGFEDataformat::TcRawData(type, ((d >> lastBit) & 0x03), 0));
        }
        if (type == hgcal::econt::STC16) {
          lastBit -= 4;
          vTc.push_back(TPGFEDataformat::TcRawData(type, ((d >> lastBit) & 0x0f), 0));
        }
      }

    } else {
      for (unsigned tc(0); tc < ECONT_FRAME::BC_MAX_TCADDR + 1; tc++) {
        if (lastBit < 1) {
          d = (d << 32);
          lastWord++;
          //if(v.size()>lastWord) d|=v[lastWord];
          d |= v[lastWord];
          lastBit += 32;
        }

        lastBit -= 1;
        if (((d >> lastBit) & 0x1) != 0) {
          unsigned tcAdd = ECONT_FRAME::BC_MAX_TCADDR - tc;  // bit map: LSB is the TC 0, MSB is TC 47
          LogDebug("[HGCalUnpackerTrigger]")
              << "index is of bit map is " << tc << " hence tc address is " << tcAdd << std::endl;
          vTc.push_back(TPGFEDataformat::TcRawData(type, tcAdd, 0));
#ifdef EDM_ML_DEBUG
          vTc.back().print();
#endif
        }
      }
      LogDebug("[HGCalUnpackerTrigger]") << "vTc.size() = " << vTc.size() << ", nTc = " << nTc << std::endl;
      if (vTc.size() != nTc) {
        throw cms::Exception("Stage1IORecoverable") << "convertElinksToTcRawData: vTc.size() != nTc\n"
                                                    << "  vTc.size() = " << vTc.size() << "\n"
                                                    << "  nTc        = " << nTc << "\n"
                                                    << "  type       = " << type;
      }
    }
    for (unsigned tc(0); tc < nTc; tc++) {
      if (lastBit < 9) {
        d = (d << 32);
        lastWord++;
        //if(v.size()>lastWord) d|=v[lastWord];
        d |= v[lastWord];
        lastBit += 32;
      }

      if (type == hgcal::econt::BestC) {
        lastBit -= 7;
        unsigned idx = nTc - tc - 1;  // to assign first energy to lowest tc address for high occupancy
        if (bitMap)
          vTc[idx] = TPGFEDataformat::TcRawData(type, vTc[idx].address(), (d >> lastBit) & 0x7f);
        else
          vTc[tc] = TPGFEDataformat::TcRawData(type, vTc[tc].address(), (d >> lastBit) & 0x7f);
      } else if (type == hgcal::econt::STC4A) {
        lastBit -= 7;
        vTc[tc] = TPGFEDataformat::TcRawData(type, vTc[tc].address(), (d >> lastBit) & 0x7f);
      } else if (type == hgcal::econt::CTC4A) {
        lastBit -= 7;
        vTc.push_back(TPGFEDataformat::TcRawData(type, tc, (d >> lastBit) & 0x7f));
      } else if (type == hgcal::econt::CTC4B) {
        lastBit -= 9;
        vTc.push_back(TPGFEDataformat::TcRawData(type, tc, (d >> lastBit) & 0x1ff));
      } else {
        lastBit -= 9;
        vTc[tc] = TPGFEDataformat::TcRawData(type, vTc[tc].address(), (d >> lastBit) & 0x1ff);
      }
    }

    //if(bx==0xf && doPrint) {
#ifdef EDM_ML_DEBUG
    LogDebug("[HGCalUnpackerTrigger]") << "TcRawData words = " << vTc.size() << std::endl;
    for (unsigned i(0); i < vTc.size(); i++) {
      vTc[i].print();
    }
#endif
  }
}  // namespace

bool HGCalUnpackerTrigger::parseFEDData(unsigned fedId,
                                        const RawFragmentWrapper& fed_data,
                                        const HGCalTriggerConfiguration& config,
                                        const HGCalMappingModuleIndexerTrigger& moduleIndexer,
                                        hgcaldigi::HGCalDigiTriggerHost& digisTrigger,
                                        hgcaldigi::HGCalFEDTriggerPacketInfoHost& fedTriggerPacketInfo,
                                        hgcaldigi::HGCalECONTPacketInfoHost& econtPacketInfo) {
  // Endianness assumption
  // From 32-bit word(ECOND) to 64-bit word(capture block): little endianness
  // Others: big endianness

  // TODO: if this also depends on the unpacking configuration, it should be moved to the specialization
  //const auto& fedConfig = config.feds[fedId];
  const auto* start_fed_data = &(fed_data.data().front());
  const auto* const header = reinterpret_cast<const uint64_t*>(start_fed_data);
  const auto* const trailer = reinterpret_cast<const uint64_t*>(start_fed_data + fed_data.size());

  uint16_t BXSLink = ((*(trailer - 1) >> BACKEND_FRAME::SLINK_BXID_POS) & BACKEND_FRAME::SLINK_BXID_MASK);

  LogDebug("[HGCalUnpackerTrigger]") << " nwords (64b) = " << std::distance(header, trailer) << "\n";

  const HGCalTriggerFedConfig& fedConfig =
      config.feds[fedId];  // NOTE: Switched to const &, we only use this to read. - Jan
  char word64[20];

#ifdef EDM_ML_DEBUG
  char num[10], word32m[20], word32l[20];
  const uint64_t* ptr = header;
  for (unsigned iword = 0; ptr < trailer; ++iword) {
    uint64_t tword = *ptr;
    uint32_t tword32m = ((tword >> TDAQ_FRAME::TDAQ_HEADER_POS) & TDAQ_FRAME::TDAQ_HEADER_MASK);
    uint32_t tword32l = tword & TDAQ_FRAME::TDAQ_HEADER_MASK;
    sprintf(num, "%03u", iword);
    sprintf(word64, "0x%016lx", tword);
    sprintf(word32m, "0x%08x", tword32m);
    sprintf(word32l, "0x%08x", tword32l);
    LogDebug("[HGCalUnpackerTrigger]") << "HGCalUnpackerTrigger::parseFEDData::tword " << num << " " << word64 << " ("
                                       << word32m << ", " << word32l << ")";
    ++ptr;
  }
#endif

  unsigned n64(std::distance(header, trailer));
  const Hgcal10gLinkReceiver::TpgSubpacketHeader* tsh(
      reinterpret_cast<const Hgcal10gLinkReceiver::TpgSubpacketHeader*>(header + 2));
  const Hgcal10gLinkReceiver::TpgSubpacketHeader* tshEnd(
      reinterpret_cast<const Hgcal10gLinkReceiver::TpgSubpacketHeader*>(header + n64 - 2 - 2));

  uint32_t econTOffset = 0;  ///THIS DEPENDS ON module
  uint32_t TdaqIdx = 0;

  bool done(false);  // bool to skip all the tdaqs > tdaqsize
  while (tsh <= tshEnd && !done) {
    const HGCalTDAQConfig& tdaqConfig =
        fedConfig.tdaqs[TdaqIdx];  // NOTE: Switched to const &, we only use this to read. - Jan

    uint32_t isValidTdaq;
    isValidTdaq = tdaqConfig.econts.size();

    LogDebug("[HGCalUnpackerTrigger]") << "tdaq idx: " << TdaqIdx << ", tdaqsize: " << isValidTdaq << std::endl;
#ifdef EDM_ML_DEBUG
    tsh->print();
#endif
    if (isValidTdaq != 0) {
      auto headerMarker = tdaqConfig.tdaqBlockHeaderMarker;
      // check header, if not valid skip the tdaq
      if (!tsh->validPattern(headerMarker)) {
        uint32_t ECONTdenseIdx = moduleIndexer.getIndexForModule(fedId, uint16_t(0));
        econtPacketInfo.view()[ECONTdenseIdx].exception() = (1 << hgcaldigi::ECONTUnpackingFlags::WrongSubpacketHeader);
        econtPacketInfo.view()[ECONTdenseIdx].location() = 0;
        econtPacketInfo.view()[ECONTdenseIdx].payloadLength() = 0;

        LogDebug("[HGCalTriggerUnpacker]")
            << "TDaq idx " << TdaqIdx << " :: Expected a header 0x" << std::hex << headerMarker << ", got 0x"
            << std::hex << tsh->pattern() << " from word = 0x" << std::hex << tsh->data() << ".";

      } else {
        if (TdaqIdx >= fedConfig.tdaqs.size()) {
          LogDebug("HGCalUnpackerTrigger") << "TDAQ index out of range for FED " << fedId << ": idx=" << TdaqIdx
                                           << ", size=" << fedConfig.tdaqs.size() << ". Skipping remaining subpackets.";
          done = true;  // To skip everything after last expected tdaq
          break;
        }

        // elinks map
        // ordering of elinks is trivial, with the 8 and 16 in the last position of the array, as default
        std::vector<uint8_t> elinks_mapping = TDAQ_FRAME::elinksDefaultMapping;

        const auto& elinksMap = fedConfig.elinksMap;
        auto it = elinksMap.find(TdaqIdx);
        // if itdaq is in the keys of elinks map, overwrite the dummy elink map
        if (it != elinksMap.end()) {
          elinks_mapping = it->second;
          LogDebug("HGCalUnpackerTrigger")
              << "Found map for re-ordering elinks on itdaq " << TdaqIdx << " and " << TdaqIdx + 1;
        }

        uint32_t nEconTs =
            isValidTdaq;  // all econts are on the first tdaq of the pair, so the second is naturally skipped in tdaq loop!

        for (unsigned bx(0); bx < tsh->numberOfBxs(); bx++) {
          LogDebug("[HGCalUnpackerTrigger]")
              << " ----------------- Start of unpacking, BX " << bx << " ---------------------------" << std::endl;
          const uint64_t* el64packed((const uint64_t*)(tsh + 1 + bx * tsh->numberOfWordsPerBx()));
          const uint32_t econTLocation =
              static_cast<uint32_t>(el64packed - header);  // should be changed, not clear how
          std::unique_ptr<uint32_t[]> elinks(
              new uint32_t[unsigned(tsh->numberOfWordsPerBx()) * 2 *
                           2]);  // allocate 16 in case of tpairs, supposing every packet has same number of words

          //filling first 7 elinks
          for (unsigned j(0); j < tsh->numberOfWordsPerBx(); j++) {
            //ordering elinks thanks to mapping
            unsigned elinkIdx = unsigned(elinks_mapping[2 * j]);
            elinks[elinkIdx] = el64packed[j] & TDAQ_FRAME::TDAQ_ELINK_MASK;
            //std::cout << "Natural order - Correct order " << 2*j + elink_offset  << " - " << elinkIdx << std::endl;
            elinkIdx = unsigned(elinks_mapping[2 * j + 1]);
            elinks[elinkIdx] = (el64packed[j] >> TDAQ_FRAME::TDAQ_ELINK_POS) & TDAQ_FRAME::TDAQ_ELINK_MASK;
            //std::cout << "Natural order - Correct order " << 2*j + elink_offset + 1 << " - " << elinkIdx  << std::endl;
            sprintf(word64, "0x%016lx", el64packed[j]);
            LogDebug("[HGCalUnpackerTrigger]")
                << "Word " << std::setw(6) << j << " = 0x" << std::hex << std::setfill('0') << std::setw(16) << word64
                << std::dec << std::setfill(' ') << std::endl;
          }

          uint8_t prevSubpacketSize = tsh->subpacketSize();
          tsh = tsh->nextSubpacketHeader();  // going to next subpacket

          if (tsh->channelId() % 2 == 0) {  // is a RX subpacket, reading toghether the next tpg subpacket (second tdaq)

            const uint64_t* el64packed2(
                (const uint64_t*)(tsh + 1 + bx * tsh->numberOfWordsPerBx()));  //second part of elinks

            // filling the remaining elinks
            for (unsigned j(0); j < tsh->numberOfWordsPerBx(); j++) {
              //ordering elinks thanks to mapping
              unsigned elinkIdx = unsigned(elinks_mapping[2 * j + TDAQ_FRAME::TDAQ_ELINK_OFFSET]);
              elinks[elinkIdx] = el64packed2[j] & TDAQ_FRAME::TDAQ_ELINK_MASK;
              //std::cout << "Natural order - Correct order " << 2*j + elink_offset  << " - " << elinkIdx << std::endl;

              elinkIdx = unsigned(elinks_mapping[2 * j + TDAQ_FRAME::TDAQ_ELINK_OFFSET + 1]);
              elinks[elinkIdx] = (el64packed2[j] >> TDAQ_FRAME::TDAQ_ELINK_POS) & TDAQ_FRAME::TDAQ_ELINK_MASK;
              //std::cout << "Natural order - Correct order " << 2*j + elink_offset + 1 << " - " << elinkIdx << std::endl;
              sprintf(word64, "0x%016lx", el64packed2[j]);
              LogDebug("[HGCalUnpackerTrigger]")
                  << "Word " << std::setw(6) << j << " = 0x" << std::hex << std::setfill('0') << std::setw(16) << word64
                  << std::dec << std::setfill(' ') << std::endl;
            }
          }
          // going back to previous subpacket, so at the next bx you always start from the first tpg of the pair
          tsh = tsh->prevSubpacketHeader(prevSubpacketSize);

          uint32_t nprevTxs = 0;
#ifdef EDM_ML_DEBUG
          for (uint32_t iel(0); iel < TDAQ_FRAME::TDAQ_PAIRS_NELINKS; iel++) {
            sprintf(word32m, "0x%08x", elinks[iel]);
            LogDebug("[HGCalUnpackerTrigger]")
                << "\t elink " << std::setw(3) << iel << " = 0x" << std::hex << std::setfill('0') << std::setw(8)
                << word32m << std::dec << std::setfill(' ') << std::endl;
          }
#endif

          for (unsigned iecon(0); iecon < nEconTs; iecon++) {
            const auto& econt_conf = tdaqConfig.econts[iecon];
            const int neTx = econt_conf.eportTxNumen;
            LogDebug("[HGCalUnpackerTrigger]") << " iecont " << iecon << " nprevTxs " << nprevTxs << std::endl;

            std::unique_ptr<uint32_t[]> el(new uint32_t[neTx]);

            // expected ECONT header from Slink BX counter
            uint16_t expEcontHeader = getExpEcontHeader(BXSLink, econt_conf.getOutType());

            LogDebug("[HGCalUnpackerTrigger]")
                << "Word :: 0x" << std::hex << (*(trailer - 1)) << std::dec << " BX counter from Slink " << BXSLink
                << " , so ECONT header should be " << expEcontHeader << std::endl;

            for (int iel = 0; iel < neTx; iel++)
              el[iel] = elinks[nprevTxs + iel];

            uint32_t econTId = iecon + econTOffset;  //unique per fedId
            uint32_t econtDenseIdx = moduleIndexer.getIndexForModule(fedId, econTId);

            if (bx == 0) {
              econtPacketInfo.view()[econtDenseIdx].exception() =
                  (1 << hgcaldigi::ECONTUnpackingFlags::NormalUnpacking);
              econtPacketInfo.view()[econtDenseIdx].location() = econTLocation;
              econtPacketInfo.view()[econtDenseIdx].payloadLength() = static_cast<uint16_t>(neTx);
            }

            TPGFEDataformat::TcRawDataPacket rdp;
            try {
              convertElinksToTcRawData(econt_conf.getOutType(), econt_conf.getNofTCs(), el.get(), rdp);
            } catch (cms::Exception& e) {
              LogDebug("Stage1IORecoverable")
                  << "BX " << bx << " Skipping ECON-T " << iecon << " (neTx=" << neTx << ")\n"
                  << e.what();
              if (bx == 0) {
                econtPacketInfo.view()[econtDenseIdx].exception() =
                    (1 << hgcaldigi::ECONTUnpackingFlags::Stage1IOConversionError);
                econtPacketInfo.view()[econtDenseIdx].location() = econTLocation;
                econtPacketInfo.view()[econtDenseIdx].payloadLength() = static_cast<uint16_t>(neTx);
              }

              continue;
            }

            LogDebug("[HGCalTriggerUnpacker]")
                << "TCs " << econt_conf.getNofTCs() << " Algo " << econt_conf.getOutType() << " econTId " << iecon
                << " Econt offset " << econTOffset << " nElinks " << econt_conf.eportTxNumen << std::endl;

// check econt bx counter on central bx is aligned with slink one
#ifdef EDM_ML_DEBUG
            if (bx == 0)
              rdp.print();
            if (bx == 3 && (rdp.bx() != expEcontHeader)) {
              LogDebug("[HGCalTriggerUnpacker]") << "ECONT header BX counter " << rdp.bx()
                                                 << " not matching expected one from Slink " << expEcontHeader;
            }
#endif

            uint32_t dropLSB = econt_conf.dropLSB;
            uint32_t totE = 0;  // module sum, for BC is over all the 48 TCs
            for (const auto& itc : rdp.getTcData())
              totE += itc.decodedE(rdp.type()) << dropLSB;

            const uint8_t algo = uint8_t(econt_conf.getOutType());
            const uint8_t nTCs = uint8_t(econt_conf.getNofTCs());
            const uint8_t sumType = uint8_t(econt_conf.sumType);

            //// How much of below will be **CONFIGURE** ed
            for (unsigned itc(0); itc < rdp.size(); itc++) {
              uint32_t tcidx = uint32_t(rdp.getTc(itc).address());
              uint32_t denseIdxRaw = moduleIndexer.getIndexForModuleData(fedId, econTId, tcidx);  // before any swapping

              // guard: payload TC address may exceed configured tcMux size (OOB read otherwise)
              if (tcidx >= econt_conf.tcMux.size()) {
                LogDebug("TrigUnpackTcMuxOOB") << "fed=" << fedId << " econTId=" << econTId << " tcidx=" << tcidx
                                               << " tcMux.size=" << econt_conf.tcMux.size();
                continue;
              }

              uint32_t maxTCsForModule = moduleIndexer.getNumChannels(fedId, econTId);
              if (econt_conf.tcMux[tcidx] >= maxTCsForModule) {
                LogDebug("TrigUnpackTcOOB") << "fed=" << fedId << " econTId=" << econTId
                                            << " TCmux=" << econt_conf.tcMux[tcidx] << " maxTC=" << maxTCsForModule;
                continue;  // Skip dummy TCs that exceed the allocated module capacity
              }

              // offset in 2 steps, first mux then econts
              int32_t tcMuxSwapOffset = econt_conf.tcMux[tcidx] - tcidx;
              int32_t econtSwapOffset = fedConfig.econtSwapOffset[iecon];
              int32_t denseIdxOffset = tcMuxSwapOffset + econtSwapOffset;

              // get offset directly from config file
              //int32_t denseIdxOffset =  econt_conf.offset[tcidx];

              uint32_t denseIdx =
                  denseIdxRaw + denseIdxOffset;  // applying offset accounting for TCs and econts swapping

              digisTrigger.view()[denseIdx].algo() = algo;
              digisTrigger.view()[denseIdx].sumType() = sumType;
              digisTrigger.view()[denseIdx].valid()(bx, 0) = true;
              digisTrigger.view()[denseIdx].nBxs() = uint8_t(tsh->numberOfBxs());
              digisTrigger.view()[denseIdx].econTId() = econTId;
              digisTrigger.view()[denseIdx].nTCs() = nTCs;
              digisTrigger.view()[denseIdx].econtHeader()(bx, 0) = uint8_t(rdp.bx());
              digisTrigger.view()[denseIdx].expEcontHeader()(bx, 0) = uint8_t(expEcontHeader);
              digisTrigger.view()[denseIdx].encodedTotE()(bx, 0) =
                  (rdp.type() == hgcal::econt::BestC) ? uint32_t(rdp.moduleSum()) : totE;
              digisTrigger.view()[denseIdx].TotE()(bx, 0) =
                  (rdp.type() == hgcal::econt::BestC)
                      ? uint32_t(TPGFEDataformat::TcRawData::Decode5E3M(rdp.moduleSum()) << dropLSB)
                      : totE;
              digisTrigger.view()[denseIdx].TCEnergy()(bx, 0) =
                  uint32_t(rdp.getTc(itc).decodedE(rdp.type()) << dropLSB);
              digisTrigger.view()[denseIdx].encodedTCEnergy()(bx, 0) = uint32_t(rdp.getTc(itc).energy());
              digisTrigger.view()[denseIdx].TCAddress()(bx, 0) = uint8_t(rdp.getTc(itc).address() + tcMuxSwapOffset);

              LogDebug("[HGCalUnpackerTrigger]")
                  << "HGCalUnpackerTrigger::parseFEDData fedId : " << fedId << ", iecon: " << iecon
                  << ", econTId: " << econTId << ", tcidx: " << tcidx << ", denseIdxRaw: " << denseIdxRaw
                  << ", tcMuxSwapOffset: " << tcMuxSwapOffset << ", econtSwapOffset: " << econtSwapOffset
                  << ", denseIdxOffset: " << denseIdxOffset << ", denseIdx: "
                  << denseIdx
                  << std::endl;
              LogDebug("[HGCalUnpackerTrigger]")
                  << "HGCalUnpackerTrigger::parseFEDData "
                  << " algo = " << uint16_t(digisTrigger.view()[denseIdx].algo())
                  << " sumType = " << uint16_t(digisTrigger.view()[denseIdx].sumType())
                  << ", valid = " << uint16_t(digisTrigger.view()[denseIdx].valid()(bx, 0))
                  << ", nBxs = " << uint16_t(digisTrigger.view()[denseIdx].nBxs())
                  << ", nTCs = " << uint16_t(digisTrigger.view()[denseIdx].nTCs())
                  << ", ieconTId = " << uint32_t(digisTrigger.view()[denseIdx].econTId()) << std::endl;
              LogDebug("[HGCalUnpackerTrigger]")
                  << "HGCalUnpackerTrigger::parseFEDData ibx : " << bx
                  << ", econt header : " << uint16_t(digisTrigger.view()[denseIdx].econtHeader()(bx, 0))
                  << ", expected econt header from Slink : "
                  << uint16_t(digisTrigger.view()[denseIdx].expEcontHeader()(bx, 0))
                  << ", MS/totE : " << uint32_t(digisTrigger.view()[denseIdx].TotE()(bx, 0)) << std::endl;
              LogDebug("[HGCalUnpackerTrigger]")
                  << "HGCalUnpackerTrigger::parseFEDData itc : " << itc
                  << ", Address: " << uint16_t(digisTrigger.view()[denseIdx].TCAddress()(bx, 0))
                  << ", Encoded Energy: " << uint32_t(digisTrigger.view()[denseIdx].encodedTCEnergy()(bx, 0))
                  << ", Unpacked Energy: " << uint32_t(digisTrigger.view()[denseIdx].TCEnergy()(bx, 0))
                  << ", Encoded MS: " << uint32_t(digisTrigger.view()[denseIdx].encodedTotE()(bx, 0))
                  << ", Unpacked MS: " << uint32_t(digisTrigger.view()[denseIdx].TotE()(bx, 0)) << std::endl;
              fedTriggerPacketInfo.view()[fedId].FEDBX_trig() = BXSLink;
              //denseIdx++;
            }  // tc loop
            //denseIndexOffset += rdp.size();
            nprevTxs += neTx;
          }  //iecon loop

        }  // bxs loop

        econTOffset += nEconTs;

      }  // if check header

    }  // if valid tdaq
    TdaqIdx++;
    tsh = tsh->nextSubpacketHeader();
  }  // tdaqs loop

  return true;
}