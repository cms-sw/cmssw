#ifndef RecoBTau_JetTagComputer_CombinedMVAV2JetTagComputer_h
#define RecoBTau_JetTagComputer_CombinedMVAV2JetTagComputer_h

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <mutex>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/TMVAEvaluator.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

class CombinedMVAV2JetTagComputer : public JetTagComputer {
    public:
        CombinedMVAV2JetTagComputer(const edm::ParameterSet &parameters);
        virtual ~CombinedMVAV2JetTagComputer();

        virtual void initialize(const JetTagComputerRecord & record) override;

        float discriminator(const TagInfoHelper &info) const override;

    private:
        std::vector<const JetTagComputer*> computers;

        const std::vector<std::string> inputComputerNames;
        mutable std::mutex m_mutex;
        [[cms::thread_guard("m_mutex")]] std::unique_ptr<TMVAEvaluator> mvaID;
};

#endif // RecoBTau_JetTagComputer_CombinedMVAV2JetTagComputer_h
