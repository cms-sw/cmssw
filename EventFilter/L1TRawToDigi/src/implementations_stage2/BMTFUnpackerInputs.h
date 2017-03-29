#include "FWCore/Framework/interface/MakerMacros.h"
#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "BMTFCollections.h"

namespace l1t{
	namespace stage2{
		struct qualityHits
		{
			int linkNo;
			int hits[3][7];
		};
		class BMTFUnpackerInputs : public Unpacker
		{
			public:
				virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
			private:
				std::map<int, qualityHits> linkAndQual_;
		};

	}
}

DEFINE_L1T_UNPACKER(l1t::stage2::BMTFUnpackerInputs);
