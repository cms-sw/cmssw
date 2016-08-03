#include "RecoLocalCalo/HcalRecAlgos/interface/parseHFPhase1AlgoDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Phase 1 HF reco algorithm headers
#include "RecoLocalCalo/HcalRecAlgos/interface/HFSimpleTimeCheck.h"

std::unique_ptr<AbsHFPhase1Algo>
parseHFPhase1AlgoDescription(const edm::ParameterSet& ps)
{
    std::unique_ptr<AbsHFPhase1Algo> algo;

    const std::string& className = ps.getParameter<std::string>("Class");

    if (className == "HFSimpleTimeCheck")
    {
        const std::vector<double>& tlimitsVec =
            ps.getParameter<std::vector<double> >("tlimits");
        const std::vector<double>& energyWeightsVec =
            ps.getParameter<std::vector<double> >("energyWeights");
        const unsigned soiPhase =
            ps.getParameter<unsigned>("soiPhase");
        const float timeShift =
            ps.getParameter<double>("timeShift");
        const float triseIfNoTDC =
            ps.getParameter<double>("triseIfNoTDC");
        const float tfallIfNoTDC =
            ps.getParameter<double>("tfallIfNoTDC");
        const bool rejectAllFailures =
            ps.getParameter<bool>("rejectAllFailures");

        std::pair<float,float> tlimits[2];
        float energyWeights[2*HFAnodeStatus::N_POSSIBLE_STATES-1][2];
        const unsigned sz = sizeof(energyWeights)/sizeof(energyWeights[0][0]);

        if (tlimitsVec.size() == 4 && energyWeightsVec.size() == sz)
        {
            tlimits[0] = std::pair<float,float>(tlimitsVec[0], tlimitsVec[1]);
            tlimits[1] = std::pair<float,float>(tlimitsVec[2], tlimitsVec[3]);

            // Same order of elements as in the natural C array mapping
            float* to = &energyWeights[0][0];
            for (unsigned i=0; i<sz; ++i)
                to[i] = energyWeightsVec[i];

            algo = std::unique_ptr<AbsHFPhase1Algo>(
                new HFSimpleTimeCheck(tlimits, energyWeights, soiPhase,
                                      timeShift, triseIfNoTDC, tfallIfNoTDC,
                                      rejectAllFailures));
        }
    }

    return algo;
}
