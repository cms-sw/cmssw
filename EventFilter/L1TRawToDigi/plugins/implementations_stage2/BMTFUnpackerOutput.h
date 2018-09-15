#include "FWCore/Framework/interface/MakerMacros.h"
#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "BMTFCollections.h"

namespace l1t{
  namespace stage2{

    class BMTFUnpackerOutput : public Unpacker
    {
    public:
      //the default constructor assumes
      //that unpacks BMTF and BMTF is triggering
      BMTFUnpackerOutput() {
	isKalman = false;
	isTriggeringAlgo = true;
      }
      BMTFUnpackerOutput(const bool isTriggering_/*, const bool isKalman_*/) {
	isTriggeringAlgo = isTriggering_;
	//isKalman = isKalman_;
      }
      ~BMTFUnpackerOutput() override{};
      bool unpack(const Block& block, UnpackerCollections *coll) override;
      void setKalmanAlgoTrue() {isKalman = true;}
    private:
      bool isTriggeringAlgo;
      bool isKalman;
    };

  }
}
