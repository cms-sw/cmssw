#include "EventFilter/HGCalRawToDigi/interface/HGCalUnpackerTrigger.h"
#include "EventFilter/HGCalRawToDigi/interface/TPG/TPGFEDataformat.hh"
#include "EventFilter/HGCalRawToDigi/interface/TPG/TPGBEDataformat.hh"
#include "EventFilter/HGCalRawToDigi/interface/TPG/Stage1IO.hh"
#include "EventFilter/HGCalRawToDigi/interface/TPG/TpgSubpacketHeader.h"
#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"
#include "DataFormats/HGCalDigi/interface/HGCalTriggerDefinitions.h"
#include "DataFormats/HGCalDigi/interface/HGCalFEDTriggerPacketInfoHost.h"
using namespace hgcal;

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

  HGCalTriggerFedConfig fedConfig = config.feds[fedId];
  const uint64_t* ptr = header;
  char num[10], word64[20], word32m[20], word32l[20];
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

  unsigned n64(std::distance(header, trailer));
  const Hgcal10gLinkReceiver::TpgSubpacketHeader* tsh(
      reinterpret_cast<const Hgcal10gLinkReceiver::TpgSubpacketHeader*>(header + 2));
  const Hgcal10gLinkReceiver::TpgSubpacketHeader* tshEnd(
      reinterpret_cast<const Hgcal10gLinkReceiver::TpgSubpacketHeader*>(header + n64 - 2 - 2));

  uint32_t econTOffset = 0;  ///THIS DEPENDS ON module
  uint32_t TdaqIdx = 0;

  bool done(false);  // bool to skip all the tdaqs > tdaqsize
  while (tsh <= tshEnd && !done) {
    HGCalTDAQConfig tdaqConfig = fedConfig.tdaqs[TdaqIdx];

    uint32_t isValidTdaq;
    isValidTdaq = tdaqConfig.econts.size();

    // std::cout << "tdaq idx: "   << TdaqIdx
    //             << ", tdaqsize: " << isValidTdaq << std::endl;

    // tsh->print();
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
        //tsh->print();

        //unsigned emp_chan(tsh->channelId()/2); // not used atm
        if (TdaqIdx >= fedConfig.tdaqs.size()) {
          LogDebug("HGCalUnpackerTrigger") << "TDAQ index out of range for FED " << fedId << ": idx=" << TdaqIdx
                                           << ", size=" << fedConfig.tdaqs.size() << ". Skipping remaining subpackets.";
          done = true;  // To skip everything after last expected tdaq
          break;
        }

        // elinks map
        // ordering of elinks is trivial, with the 8 and 16 in the last position of the array, as default
        std::vector<uint8_t> elinks_mapping = {0, 1, 2, 3, 4, 5, 6, 14, 7, 8, 9, 10, 11, 12, 13, 15};

        auto elinksMap = fedConfig.elinksMap;
        // if itdaq is in the keys of elinks map, overwrite the dummy elink map
        if (elinksMap.find(TdaqIdx) != elinksMap.end()) {
          elinks_mapping = elinksMap[TdaqIdx];
          LogDebug("HGCalUnpackerTrigger")
              << "Found map for re-ordering elinks on itdaq " << TdaqIdx << " and " << TdaqIdx + 1;
        }

        uint32_t nEconTs =
            isValidTdaq;  // all econts are on the first tdaq of the pair, so the second is naturally skipped in tdaq loop!

        for (unsigned bx(0); bx < tsh->numberOfBxs(); bx++) {
          //std::cout << " ----------------- Start of unpacking, BX " << bx << " ---------------------------" << std::endl;
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

          std::unique_ptr<uint16_t[]> S1Tcs(
              new uint16_t[15]);  // allocate 15 to handle tiles as well (6 + 9 TCs), to read stage I output

          //std::cout << "So its a " << tsh->channelId()%2 << std::endl;
          if (tsh->channelId() % 2 == 0) {  // is a RX subpacket, reading toghether the next tpg subpacket (second tdaq)

            //std::cout << " going to next subpacket for remaining 7 elinks " << std::endl;
            //tsh->print();
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
            // going back to previous subpacket, so at the next bx you always start from the first tpg of the pair
            tsh = tsh->prevSubpacketHeader(prevSubpacketSize);
            //std::cout << " going back to first subpacket" << std::endl;
            //tsh->print();

          } else {  // is a TX subpacket, reading stage I
            // -----  now reading the TX
            //std::cout << "I have read RX " << std::endl;
            //tsh->print();

            //std::cout << "NOW reading the TX ... and here we are..." << std::endl;
            //tsh->print();
            const uint64_t* S164bitword((const uint64_t*)(tsh + 1 + bx * tsh->numberOfWordsPerBx()));
            //uint16_t msStageI = 0;
            //uint16_t msStageI2 = 0;

            //filling StageI output (TX channels) // all very hardcoded
            for (unsigned j(0); j < 6; j++) {
              //msStageI = ((S164bitword[0] & 0xffff) >> 6 ) & (0x7f) ;
              if (j < 5)
                S1Tcs[j] = S164bitword[j + 1] & 0xffff;  //first column
              if (j < 4)
                S1Tcs[j + 5] = (S164bitword[j + 1] >> 16) & 0xffff;  //second column
              if (nEconTs > 1) {
                S1Tcs[j + 9] = (S164bitword[j + 1] >> 32) & 0xffff;  //third column (only for tiles)
                                                                     //msStageI2 = (S164bitword[0]>>32) & 0xffff;
                                                                     //msStageI2 = (msStageI2 >> 6 ) & (0x7f) ;
              } else
                S1Tcs[j + 9] = 0;
            }
            // if (bx == 3){
            //     std::cout << "Module SUM: " << msStageI << std::endl;
            //     if (nEconTs > 1) std::cout << "Module SUM: " << msStageI2 << std::endl;
            //     for(unsigned itc(0);itc<15;itc++) {
            //         std::cout << "\t Tcs " << std::setw(3) << itc << " = 0x"
            //    			      << std::hex << std::setfill('0')
            //   			      << std::setw(4) << S1Tcs[itc]
            //   			      << std::dec << " TC: "<< ( S1Tcs[itc] & (0x3f))<< std::setfill(' ')
            // 			      << std::dec << " En: "<< (( S1Tcs[itc] >> 6) & (0x1ff))<< std::setfill(' ')
            //   			      << std::endl;
            //     }
            // }
            // std::cout << "I am done... getting back to RX" << std::endl;

            tsh = tsh->prevSubpacketHeader(prevSubpacketSize);
            //tsh->print();
          }

          uint32_t nprevTxs = 0;
          for (uint32_t iel(0); iel < TDAQ_FRAME::TDAQ_PAIRS_NELINKS; iel++) {
            sprintf(word32m, "0x%08x", elinks[iel]);
            LogDebug("[HGCalUnpackerTrigger]")
                << "\t elink " << std::setw(3) << iel << " = 0x" << std::hex << std::setfill('0') << std::setw(8)
                << word32m << std::dec << std::setfill(' ') << std::endl;
          }

          for (unsigned iecon(0); iecon < nEconTs; iecon++) {
            const auto& econt_conf = tdaqConfig.econts[iecon];
            const int neTx = econt_conf.eportTxNumen;
            // std::cout << "iecont " << iecon << std::endl;
            // std::cout << "nprevTxs " << nprevTxs << std::endl;
            // std::cout << "neTx " << neTx << std::endl;
            std::unique_ptr<uint32_t[]> el(new uint32_t[neTx]);
            TPGFEConfiguration::ConfigEconT cfgecont;
            cfgecont.setNElinks(uint32_t(neTx));

            const int select = econt_conf.select;
            //std::cout << "select " << select << std::endl;
            cfgecont.setSelect(select);

            const int dropLSB = econt_conf.dropLSB;
            cfgecont.setDropLSB(dropLSB);

            //std::cout << "dropLSB from cfg: " << dropLSB << " ECONT dropLSB set to: " << cfgecont.getDropLSB() << std::endl;

            const bool sumType = econt_conf.sumType;
            cfgecont.setMSSumType(sumType);

            // std::cout << "sumType from cfg: " << sumType << " ECONT sumType set to: " << cfgecont.getMSSumType() << std::endl;

            // expected ECONT header from Slink BX counter
            uint16_t expEcontHeader = TPGFEConfiguration::getExpEcontHeader(BXSLink, cfgecont.getOutType());

            LogDebug("[HGCalUnpackerTrigger]")
                << "Word :: 0x" << std::hex << (*(trailer - 1)) << std::dec << " BX counter from Slink " << BXSLink
                << " , so ECONT header should be " << expEcontHeader << std::endl;

            for (int iel = 0; iel < neTx; iel++)
              el[iel] = elinks[nprevTxs + iel];

            uint32_t econTId = iecon + econTOffset;  //unique per fedId
            uint32_t econtDenseIdx = moduleIndexer.getIndexForModule(fedId, econTId);
            //std::cout <<  "ECONT dense "<< econtDenseIdx << " econtid " << econTId << std::endl;
            if (bx == 0) {
              econtPacketInfo.view()[econtDenseIdx].exception() =
                  (1 << hgcaldigi::ECONTUnpackingFlags::NormalUnpacking);
              econtPacketInfo.view()[econtDenseIdx].location() = econTLocation;
              econtPacketInfo.view()[econtDenseIdx].payloadLength() = static_cast<uint16_t>(neTx);
            }

            TPGFEDataformat::TcRawDataPacket rdp;
            try {
              TPGStage1Emulation::Stage1IO::convertElinksToTcRawData(
                  cfgecont.getOutType(), cfgecont.getNofTCs(), el.get(), rdp);
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

            //if (bx == 0) rdp.print();
            //std::cout <<  "TCs "<< cfgecont.getNofTCs() <<  " out "<< cfgecont.getOutType() << " econTId " << iecon << " offset "  << econTOffset << " nElinks "<< cfgecont.getNElinks() << " Select " << cfgecont.getSelect()  << std::endl;

            // check econt bx counter on central bx is aligned with slink one
            if (bx == 3 && (rdp.bx() != expEcontHeader)) {
              LogDebug("[HGCalTriggerUnpacker]") << "ECONT header BX counter " << rdp.bx()
                                                 << " not matching expected one from Slink " << expEcontHeader;
            }

            uint32_t totE = 0;  // module sum, for BC is over all the 48 TCs
            for (const auto& itc : rdp.getTcData())
              totE += itc.decodedE(rdp.type()) >> cfgecont.getDropLSB();

            //// How much of below will be **CONFIGURE** ed
            for (unsigned itc(0); itc < rdp.size(); itc++) {
              //uint32_t tcidx = uint32_t(rdp.getTc(itc).address());
              uint32_t tcidx = itc;
              //uint32_t denseIdx = tcidx + fedReadoutSequence.TCOffsets_.at(econTId) ; //same as following function call
              uint32_t denseIdxRaw = moduleIndexer.getIndexForModuleData(fedId, econTId, tcidx);  // before any swapping

              // offset in 2 steps, first mux then econts
              uint32_t tcMuxSwapOffset = econt_conf.tcMux[itc] - tcidx;
              uint32_t econtSwapOffset = fedConfig.econtSwapOffset[iecon];
              uint32_t denseIdxOffset = tcMuxSwapOffset + econtSwapOffset;

              // get offset directly from config file
              //uint32_t denseIdxOffset =  econt_conf.offset[itc];

              uint32_t denseIdx =
                  denseIdxRaw + denseIdxOffset;  // applying offset accounting for TCs and econts swapping

              digisTrigger.view()[denseIdx].algo() = uint8_t(cfgecont.getOutType());
              digisTrigger.view()[denseIdx].sumType() = uint8_t(cfgecont.getMSSumType());
              digisTrigger.view()[denseIdx].valid()(bx, 0) = true;
              digisTrigger.view()[denseIdx].nBxs() = uint8_t(tsh->numberOfBxs());
              digisTrigger.view()[denseIdx].econTId() = econTId;
              digisTrigger.view()[denseIdx].nTCs() = uint8_t(cfgecont.getNofTCs());
              digisTrigger.view()[denseIdx].econtHeader()(bx, 0) = uint8_t(rdp.bx());
              digisTrigger.view()[denseIdx].expEcontHeader()(bx, 0) = uint8_t(expEcontHeader);
              digisTrigger.view()[denseIdx].encodedTotE()(bx, 0) =
                  (rdp.type() == TPGFEDataformat::BestC) ? uint32_t(rdp.moduleSum()) : totE;
              digisTrigger.view()[denseIdx].TotE()(bx, 0) =
                  (rdp.type() == TPGFEDataformat::BestC)
                      ? uint32_t(TPGFEDataformat::TcRawData::Decode5E3M(rdp.moduleSum()))
                      : totE;
              digisTrigger.view()[denseIdx].TCEnergy()(bx, 0) =
                  uint32_t(rdp.getTc(itc).decodedE(rdp.type()) << cfgecont.getDropLSB());
              digisTrigger.view()[denseIdx].encodedTCEnergy()(bx, 0) = uint32_t(rdp.getTc(itc).energy());
              digisTrigger.view()[denseIdx].TCAddress()(bx, 0) = uint8_t(rdp.getTc(itc).address());

              //if (bx == 3 && (tsh->nextSubpacketHeader())->channelId()%2!=0 ) {
              if (bx == 3) {
                econtPacketInfo.view()[econtDenseIdx].nTCs() = uint8_t(cfgecont.getNofTCs());
                if (econTId != 10)
                  econtPacketInfo.view()[econtDenseIdx].TCEnergy_Stage1()(0, itc) =
                      uint16_t(((S1Tcs[itc] >> 6) & (0x1ff)));  // getting the decoded energy
                else
                  econtPacketInfo.view()[econtDenseIdx].TCEnergy_Stage1()(0, itc) =
                      uint16_t(((S1Tcs[itc + 9] >> 6) & (0x1ff)));  //for the special case of the second tile

                // std::cout   << "Econt idx " << econTId
                //             << " MS encoded " <<  digisTrigger.view()[denseIdx].encodedTotE()(bx,0)
                //             << " MS " <<  digisTrigger.view()[denseIdx].TotE()(bx,0)
                //             << " TC Addr : "<< uint16_t(digisTrigger.view()[denseIdx].TCAddress()(bx,0))
                //             << " encoded En: " <<  uint16_t(digisTrigger.view()[denseIdx].encodedTCEnergy()(bx,0))
                //             << " EN: " <<  uint16_t(digisTrigger.view()[denseIdx].TCEnergy()(bx,0))
                // 		    << " stage I En: " <<  uint16_t(econtPacketInfo.view()[econtDenseIdx].TCEnergy_Stage1()(0,itc))
                // 		    << std::endl;
              }

              LogDebug("[HGCalUnpackerTrigger]")
                  << "HGCalUnpackerTrigger::parseFEDData fedId : " << fedId << ", iecon: " << iecon
                  << ", econTId: " << econTId << ", tcidx: " << tcidx << ", denseIdxRaw: " << denseIdxRaw
                  << ", tcMuxSwapOffset: " << tcMuxSwapOffset << ", econtSwapOffset: " << econtSwapOffset
                  << ", denseIdxOffset: " << denseIdxOffset << ", denseIdx: "
                  << denseIdx
                  //  << ", getDenseTCIndex00: " << moduleIndexer.getDenseTCIndex(fedId, econTId, 0, tcidx)
                  //  << ", getDenseTCIndex01: " << moduleIndexer.getDenseTCIndex(fedId, econTId+1, 1, tcidx)
                  //  << ", getDenseTCIndex02: " << moduleIndexer.getDenseTCIndex(fedId, econTId+2, 2, tcidx)
                  //  << ", getIndexForModuleData00: " << moduleIndexer.getIndexForModuleData(fedId, econTId, tcidx)
                  //  << ", getIndexForModuleData01: " << moduleIndexer.getIndexForModuleData(fedId, econTId+1, tcidx)
                  //  << ", getIndexForModuleData02: " << moduleIndexer.getIndexForModuleData(fedId, econTId+2, tcidx)
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

bool HGCalUnpackerTrigger::parseTDAQBlock() { return true; }
