#include "FWCore/Framework/interface/MakerMacros.h"
#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "BMTFCollections.h"

namespace l1t{
  namespace stage2{

    class BMTFUnpackerOutput : public Unpacker
    {
    public:
      //the constructors assume that unpacks
      //BMTF and BMTF is triggering (except assigned differently)
      BMTFUnpackerOutput() {
	isKalman = false;
	isTriggeringAlgo = true;
      }
      BMTFUnpackerOutput(const bool isTriggering_) {
	isKalman = false;
	isTriggeringAlgo = isTriggering_;
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
