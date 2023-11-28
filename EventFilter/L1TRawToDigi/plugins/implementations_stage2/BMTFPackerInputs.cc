#include "BMTFPackerInputs.h"

#include <vector>
//#include <bitset>//debug

namespace l1t {
  namespace stage2 {

    const int BMTFPackerInputs::ownLinks_[] = {4, 5, 12, 13, 20, 21, 22, 23, 28, 29};

    Blocks BMTFPackerInputs::pack(const edm::Event& event, const PackerTokens* toks) {
      int board_id = (int)board();

      auto muonPhToken = static_cast<const BMTFTokens*>(toks)->getInputMuonTokenPh();
      auto muonThToken = static_cast<const BMTFTokens*>(toks)->getInputMuonTokenTh();

      Blocks blocks;

      edm::Handle<L1MuDTChambPhContainer> phInputs;
      event.getByToken(muonPhToken, phInputs);
      edm::Handle<L1MuDTChambThContainer> thInputs;
      event.getByToken(muonThToken, thInputs);

      uint32_t qualEta_32bit = 0;
      uint32_t posEta_0_32bit = 0;
      uint32_t posEta_n2_32bit = 0, posEta_n1_32bit = 0;
      uint32_t posEta_p2_32bit = 0, posEta_p1_32bit = 0;

      bool moreBXeta = false;
      for (int link = 0; link <= 35; link++) {
        //	  std::cout << "link : " << link << std::endl;

        if ((link >= 6 && link < 8) || (link >= 14 && link < 16) || (link >= 30 && link < 32))
          continue;

        //initializing null block_payloads and block_id
        std::vector<uint32_t> payload_0(6, 0);
        std::vector<uint32_t> payload_p1(6, 0);
        std::vector<uint32_t> payload_n1(6, 0);
        std::vector<uint32_t> payload_p2(6, 0);
        std::vector<uint32_t> payload_n2(6, 0);

        unsigned int block_id = 2 * link;

        std::vector<bool> bxPresent(5, false);
        bool moreBXphi = false;

        unsigned int BC = 0;

        //The first 4 phi words for the link's payload

        for (const auto& iphi : *(phInputs->getContainer())) {
          // Only allow -2 <= bxNum <= 2, as in Stage2 data
          if (std::abs(iphi.bxNum()) > 2)
            continue;

          if (iphi.bxNum() != 0)
            moreBXphi = true;

          //	      std::cout << "scNum+1 = " << iphi->scNum()+1 << ",   board_id = " << board_id << std::endl;
          // 	      std::cout << "bx, station = " << iphi->bxNum() << ", " << iphi->stNum() << std::endl;

          //BC = iphi->BxCnt();//this thing here is not completely functional

          if (iphi.scNum() + 1 != board_id)
            continue;
          //	      std::cout << "correct board" << std::endl;

          if (link != ownLinks_[4 + 2 * (iphi.whNum()) + iphi.Ts2Tag()])
            continue;
          // 	      std::cout << "correct link" << std::endl;

          bxPresent[2 + iphi.bxNum()] = true;

          //1 create 32word, 2 insert 32word in correct Block Slot
          uint32_t word_32bit = wordPhMaker(iphi);  //1
          if (bxPresent[0]) {
            payload_n2[iphi.stNum() - 1] = word_32bit;
          } else if (bxPresent[1]) {
            payload_n1[iphi.stNum() - 1] = word_32bit;
          } else if (bxPresent[2])
            payload_0[iphi.stNum() - 1] = word_32bit;
          else if (bxPresent[3]) {
            payload_p1[iphi.stNum() - 1] = word_32bit;
          } else if (bxPresent[4]) {
            payload_p2[iphi.stNum() - 1] = word_32bit;
          }

          bxPresent.assign(5, false);

        }  //phiCont_itr

        //============================================================================================

        //Create phiNull words (use the old format for null words because Packer gives fw_id=1)
        //uint32_t phiNull_32bit = 0 | (BC & 0x3) << 30 | (7 & 0x7) << 22; //for the new fw
        uint32_t phiNull_32bit = 0 | (BC & 0x3) << 30;  //for the old fw
        //phiNull = (BC)000001110000000000000000000000
        uint32_t etaNull_32bit = 0 | (BC & 0x3) << 30;
        //etaNull = (BC)000000000000000000000000000000

        //============================================================================================

        //The 5th & 6th words of the link's payload
        //	  std::cout << "link%2 = " << link%2 << std::endl;
        if (link % 2 == 0) {
          //these Eta vars have to be declared out of link itr scope in order to maintain its information for the next link
          //Using these as the basis for the pos and qual 32bit eta word
          //in case there are les than 3 hits, the entries will be zero.
          posEta_0_32bit = etaNull_32bit;
          posEta_n2_32bit = etaNull_32bit;
          posEta_n1_32bit = etaNull_32bit;
          posEta_p2_32bit = etaNull_32bit;
          posEta_p1_32bit = etaNull_32bit;

          qualEta_32bit = etaNull_32bit;

          for (const auto& ithe : *(thInputs->getContainer())) {
            // Only allow -2 <= bxNum <= 2, as in Stage2 data
            if (std::abs(ithe.bxNum()) > 2)
              continue;

            if (ithe.bxNum() != 0)
              moreBXeta = true;

            //debug
            //		std::cout << "scNum+1 = " << ithe.scNum()+1 << ",   board_id = " << board_id << std::endl;
            //		std::cout << "bx, station = " << ithe.bxNum() << ", " << ithe.stNum() << std::endl;
            //		std::cout << "related link: " << ownLinks_[4+2*(ithe.whNum())] << std::endl;

            if (ithe.scNum() + 1 != board_id)
              continue;

            if (link != ownLinks_[4 + 2 * (ithe.whNum())])
              continue;

            bxPresent[2 + ithe.bxNum()] = true;

            //positions for the next link
            uint32_t posEta_7bit = wordThMaker(ithe, false);

            //qualities for this link
            uint32_t qualEta_7bit = wordThMaker(ithe, true);
            qualEta_32bit = qualEta_32bit | ((qualEta_7bit & 0x7F) << 7 * (ithe.stNum() - 1));

            //write the eta-pos and eta-qual information at the correct payload per BX
            if (bxPresent[0]) {
              payload_n2[4] = qualEta_32bit;
              payload_n2[5] = etaNull_32bit | (2 & 0x2);
              posEta_n2_32bit = posEta_n2_32bit | ((posEta_7bit & 0x7F) << 7 * (ithe.stNum() - 1));
            } else if (bxPresent[1]) {
              payload_n1[4] = qualEta_32bit;
              payload_n1[5] = etaNull_32bit | (2 & 0x2);
              posEta_n1_32bit = posEta_n1_32bit | ((posEta_7bit & 0x7F) << 7 * (ithe.stNum() - 1));
            } else if (bxPresent[2]) {
              payload_0[4] = qualEta_32bit;
              payload_0[5] = etaNull_32bit | (2 & 0x2);
              posEta_0_32bit = posEta_0_32bit | ((posEta_7bit & 0x7F) << 7 * (ithe.stNum() - 1));
            } else if (bxPresent[3]) {
              payload_p1[4] = qualEta_32bit;
              payload_p1[5] = etaNull_32bit | (2 & 0x2);
              posEta_p1_32bit = posEta_p1_32bit | ((posEta_7bit & 0x7F) << 7 * (ithe.stNum() - 1));
            } else if (bxPresent[4]) {
              payload_p2[4] = qualEta_32bit;
              payload_p2[5] = etaNull_32bit | (2 & 0x2);
              posEta_p2_32bit = posEta_p2_32bit | ((posEta_7bit & 0x7F) << 7 * (ithe.stNum() - 1));
            }

            bxPresent.assign(5, false);

          }  //theCont_itr

        } else {  //now that we are in the next prime link #, write the buffered eta-qual

          if (moreBXeta) {
            payload_n2[4] = posEta_n2_32bit;
            payload_n2[5] = etaNull_32bit;

            payload_n1[4] = posEta_n1_32bit;
            payload_n1[5] = etaNull_32bit;
          }

          payload_0[4] = posEta_0_32bit;
          payload_0[5] = etaNull_32bit;

          if (moreBXeta) {
            payload_p1[4] = posEta_p1_32bit;
            payload_p1[5] = etaNull_32bit;

            payload_p2[4] = posEta_p2_32bit;
            payload_p2[5] = etaNull_32bit;
          }
        }

        //std::cout << "moreBXphi" << moreBXphi << std::endl;
        //std::cout << "moreBXeta" << moreBXeta << std::endl;

        bool moreBX = moreBXphi || moreBXeta;
        //std::cout << "moreBX" << moreBX << std::endl;

        //case where phi words are notcreated
        for (int iSt = 0; iSt <= 3; iSt++) {
          if (moreBX && payload_n2[iSt] == 0)
            payload_n2[iSt] = phiNull_32bit;

          if (moreBX && payload_n1[iSt] == 0)
            payload_n1[iSt] = phiNull_32bit;

          if (payload_0[iSt] == 0)
            payload_0[iSt] = phiNull_32bit;

          if (moreBX && payload_p1[iSt] == 0)
            payload_p1[iSt] = phiNull_32bit;

          if (moreBX && payload_p2[iSt] == 0)
            payload_p2[iSt] = phiNull_32bit;
        }

        //case where eta words are notcreated
        for (int word = 4; word <= 5; word++) {
          if (moreBX && payload_n2[word] == 0)
            payload_n2[word] = etaNull_32bit;

          if (moreBX && payload_n1[word] == 0)
            payload_n1[word] = etaNull_32bit;

          if (payload_0[word] == 0)
            payload_0[word] = etaNull_32bit;

          if (moreBX && payload_p1[word] == 0)
            payload_p1[word] = etaNull_32bit;

          if (moreBX && payload_p2[word] == 0)
            payload_p2[word] = etaNull_32bit;
        }

        //============================================================================================

        /*

	  //debug
	  std::cout << "payload created : " << std::endl;
	  if (moreBX){	    
	    for (auto &word : payload_n2)	// 
	      std::cout << std::bitset<32>(word).to_string() << std::endl;
	  
	    for (auto &word : payload_n1)	// 
	      std::cout << std::bitset<32>(word).to_string() << std::endl;
	  }
	  
	  for (auto &word : payload_0)	// 
	    std::cout << std::bitset<32>(word).to_string() << std::endl;
	  
	  if (moreBX){
	    for (auto &word : payload_p1)	// 
	      std::cout << std::bitset<32>(word).to_string() << std::endl;
	  
	    for (auto &word : payload_p2)	// 
	      std::cout << std::bitset<32>(word).to_string() << std::endl;
	  }
	  */
        //============================================================================================

        std::vector<uint32_t> payload;

        if (moreBX) {  //push -2,-1 bx payloads
          for (int i = 0; i < 6; i++)
            payload.push_back(payload_n2[i]);
          for (int i = 0; i < 6; i++)
            payload.push_back(payload_n1[i]);
        }

        for (int i = 0; i < 6; i++)  //zero bx payload
          payload.push_back(payload_0[i]);

        if (moreBX) {  //push +1,+2 bx payloads
          for (int i = 0; i < 6; i++)
            payload.push_back(payload_p1[i]);
          for (int i = 0; i < 6; i++)
            payload.push_back(payload_p2[i]);
        }

        //in format Block(id,payload)
        blocks.push_back(Block(block_id, payload));

        if (link % 2 != 0) {
          moreBXeta = false;
        }

      }  //link_itr

      return blocks;
    }

    uint32_t BMTFPackerInputs::wordPhMaker(const L1MuDTChambPhDigi& phInput) {
      uint32_t temp(0);

      temp = (phInput.phi() & phiMask) << phiShift | (phInput.phiB() & phiBMask) << phiBShift |
             (phInput.code() & qualMask) << qualShift | (phInput.RpcBit() & rpcMask) << rpcShift | (0) << 29 |
             (phInput.BxCnt() & bxCntMask) << bxCntShift;

      //	std::cout<<"uint32_t temp is: "<<std::bitset<32>(temp).to_string()<<"---- phi="<<phInput.phi()<<", phiB()="<<phInput.phiB()<<", code(qual?)="<<phInput.code()<<", RPC="<<phInput.RpcBit()<<", BC="<<phInput.BxCnt()<<std::endl;
      return temp;
    }

    uint32_t BMTFPackerInputs::wordThMaker(const L1MuDTChambThDigi& thInput, const bool& qualFlag) {
      uint32_t temp(0);
      if (!qualFlag) {
        for (int i = 6; i >= 0; i--) {
          temp = temp << 1;
          temp = temp | (thInput.position(i) & 0x1);
        }
      } else {
        for (int i = 6; i >= 0; i--) {
          temp = temp << 1;
          temp = temp | (thInput.quality(i) & 0x1);
        }
      }

      return temp;
    }

  }  // namespace stage2
}  // namespace l1t
DEFINE_L1T_PACKER(l1t::stage2::BMTFPackerInputs);
