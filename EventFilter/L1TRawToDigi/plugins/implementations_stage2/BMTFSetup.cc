#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/PackingSetupFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "BMTFSetup.h"

namespace l1t {
  namespace stage2 {

    std::unique_ptr<PackerTokens> BMTFSetup::registerConsumes(const edm::ParameterSet& cfg,
                                                              edm::ConsumesCollector& cc) {
      return std::unique_ptr<PackerTokens>(new BMTFTokens(cfg, cc));
    }

    void BMTFSetup::fillDescription(edm::ParameterSetDescription& desc){};

    PackerMap BMTFSetup::getPackers(int fed, unsigned int fw) {
      PackerMap res;
      //res are in format res[amc_no, board_id]

      if (fed == 1376 || fed == 1377) {
        std::array<int, 12> board_out = {{1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12}};  //these are board_ids per amc_no-1

        for (unsigned int i = 1; i <= board_out.size(); i++) {
          auto packer_out = std::make_shared<BMTFPackerOutput>();
          auto packer_in = PackerFactory::get()->make("stage2::BMTFPackerInputs");
          if (fw >= 2452619552) {  // the 1st Kalman fw-ver value (0x95000160)
            packer_out->setKalmanAlgoTrue();
          }
          res[{i, board_out[i - 1]}] = {packer_out, packer_in};
        }
      }  //if BMTF feds

      return res;
    }  //getPackers

    void BMTFSetup::registerProducts(edm::ProducesCollector prod) {
      prod.produces<RegionalMuonCandBxCollection>("BMTF");
      prod.produces<RegionalMuonCandBxCollection>("BMTF2");
      prod.produces<L1MuDTChambPhContainer>();
      prod.produces<L1MuDTChambThContainer>();

      // Depricated
      //prod.produces<L1MuDTChambPhContainer>("PhiDigis");
      //prod.produces<L1MuDTChambThContainer>("TheDigis");
    }

    std::unique_ptr<UnpackerCollections> BMTFSetup::getCollections(edm::Event& e) {
      return std::unique_ptr<UnpackerCollections>(new BMTFCollections(e));
    }

    UnpackerMap BMTFSetup::getUnpackers(int fed, int board, int amc, unsigned int fw) {
      auto inputMuonsOld = UnpackerFactory::get()->make("stage2::BMTFUnpackerInputsOldQual");
      auto inputMuonsNew = UnpackerFactory::get()->make("stage2::BMTFUnpackerInputsNewQual");

      auto outputMuon = std::make_shared<BMTFUnpackerOutput>();        //here is the triggering collection
      auto outputMuon2 = std::make_shared<BMTFUnpackerOutput>(false);  //here is the secondary
      if (fw >= 2499805536)                                            //this is in HEX '95000160'
        outputMuon->setKalmanAlgoTrue();
      else
        outputMuon2->setKalmanAlgoTrue();

      UnpackerMap res;
      if (fed == 1376 || fed == 1377) {
        // Input links
        for (int iL = 0; iL <= 70; iL += 2) {
          if (iL == 12 || iL == 14 || (iL > 26 && iL < 32) || iL == 60 || iL == 62)
            continue;

          if (fw < 2452619552) {
            res[iL] = inputMuonsOld;
          } else {
            res[iL] = inputMuonsNew;
          }
        }

        // Output links
        res[123] = outputMuon;
        res[125] = outputMuon2;
      }

      return res;
    };
  };  // namespace stage2
}  // namespace l1t

DEFINE_L1T_PACKING_SETUP(l1t::stage2::BMTFSetup);
