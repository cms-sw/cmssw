#ifndef RecoLocalCalo_EcalDeadChannelRecoveryAlgos_DeadChannelNNContext_H
#define RecoLocalCalo_EcalDeadChannelRecoveryAlgos_DeadChannelNNContext_H

#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/xyNNEB.h"
#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/xyNNEE.h"

#include <functional>

class DeadChannelNNContext {
    public:
		DeadChannelNNContext();
		~DeadChannelNNContext();

        enum NetworkID { 
          ccEB = 0, ccEE,
          rrEB, rrEE,
          llEB, llEE,
                     
          uuEB, uuEE,
          ddEB, ddEE,
          ruEB, ruEE,
                     
          rdEB, rdEE,
          luEB, luEE,
          ldEB, ldEE,
        };

        double value(NetworkID method, int index, double in0, double in1, double in2, double in3, double in4, double in5, double in6, double in7);

    private:
		void load();
        typedef std::function<double(double[8])> NNFunc3x3;
        NNFunc3x3 implementation[18];
};

#endif
