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
          if (i % 2 != 0) {  //maybe this check is not needed
            res[{i, board_out[i - 1]}] = {PackerFactory::get()->make("stage2::BMTFPackerOutput"),
                                          PackerFactory::get()->make("stage2::BMTFPackerInputs")};
          } else {
            res[{i, board_out[i - 1]}] = {PackerFactory::get()->make("stage2::BMTFPackerOutput"),
                                          PackerFactory::get()->make("stage2::BMTFPackerInputs")};
          }
        }
      }  //if feds

      /*
	 if (fed == 1376) {
	   std::cout << "fed is 1376" << std::endl;
	   for (int i=1; i <= 12; i = i+2){//itr for amc_no = 1,3,5,7,9,11
	     res[{i,board_out[i-1]}] = {PackerFactory::get()->make("stage2::BMTFPackerOutput"),
					PackerFactory::get()->make("stage2::BMTFPackerInputs")};
	   }

	 }
	 else if (fed == 1377) {
	   std::cout << "fed is 1377" << std::endl;
	   for (int i=2; i <=12; i = i+2){//itr for amc_no = 2,4,6,8,10,12
	     res[{i,board_out[i-1]}] = {PackerFactory::get()->make("stage2::BMTFPackerOutput"),
					PackerFactory::get()->make("stage2::BMTFPackerInputs")};
	   }
	
	 }
	 else{
	   std::cout << std::endl;
	   std::cout << "The given fed is not a BMTF fed (1376 or 1377)" << std::endl;
	   std::cout << std::endl;
	 }//if feds
	   */

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
