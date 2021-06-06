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
        for (auto board : boardIdPerSlot) {
          auto packer_out = std::make_shared<BMTFPackerOutput>();
          auto packer_in = PackerFactory::get()->make("stage2::BMTFPackerInputs");
          if (fw >= firstKalmanFwVer) {
            packer_out->setKalmanAlgoTrue();
          }
          res[{board.first, board.second}] = {packer_out, packer_in};
        }
      }  //if BMTF feds

      return res;
    }  //getPackers

    void BMTFSetup::registerProducts(edm::ProducesCollector prod) {
      prod.produces<RegionalMuonCandBxCollection>("BMTF");
      prod.produces<RegionalMuonCandBxCollection>("BMTF2");
      prod.produces<L1MuDTChambPhContainer>();
      prod.produces<L1MuDTChambThContainer>();
    }

    std::unique_ptr<UnpackerCollections> BMTFSetup::getCollections(edm::Event& e) {
      return std::unique_ptr<UnpackerCollections>(new BMTFCollections(e));
    }

    UnpackerMap BMTFSetup::getUnpackers(int fed, int board, int amc, unsigned int fw) {
      auto inputMuonsOld = UnpackerFactory::get()->make("stage2::BMTFUnpackerInputsOldQual");
      auto inputMuonsNew = UnpackerFactory::get()->make("stage2::BMTFUnpackerInputsNewQual");
      auto outputMuon = std::make_shared<BMTFUnpackerOutput>();        // triggering collection
      auto outputMuon2 = std::make_shared<BMTFUnpackerOutput>(false);  // secondary coll
      if (fw >= firstKalmanFwVer)
        outputMuon->setKalmanAlgoTrue();
      else
        outputMuon2->setKalmanAlgoTrue();

      UnpackerMap res;
      if (fed == 1376 || fed == 1377) {
        // Input links
        for (int iL = 0; iL <= 70; iL += 2) {
          if (iL == 12 || iL == 14 || (iL > 26 && iL < 32) || iL == 60 || iL == 62)
            continue;

          if (fw < firstNewInputsFwVer) {
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
