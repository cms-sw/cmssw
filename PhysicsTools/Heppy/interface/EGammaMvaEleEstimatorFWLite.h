#ifndef PhysicsTools_Heppy_EGammaMvaEleEstimatorFWLite_h
#define PhysicsTools_Heppy_EGammaMvaEleEstimatorFWLite_h

struct EGammaMvaEleEstimator;
struct EGammaMvaEleEstimatorCSA14;
namespace reco { struct Vertex; }
namespace pat { struct Electron; }
#include <vector>
#include <string>

namespace heppy {

class EGammaMvaEleEstimatorFWLite {
    public:
        EGammaMvaEleEstimatorFWLite();
        ~EGammaMvaEleEstimatorFWLite();

        enum MVAType {
            kTrig = 0, // MVA for triggering electrons
            kTrigNoIP = 1, // MVA for triggering electrons without IP info
            kNonTrig = 2, // MVA for non-triggering electrons 
            kTrigCSA14 = 3, // MVA for non-triggering electrons 
            kNonTrigCSA14 = 4, // MVA for non-triggering electrons 
            kNonTrigPhys14 = 5, // MVA for non-triggering electrons 
        };

        void initialize( std::string methodName,
                MVAType type,
                bool useBinnedVersion,
                std::vector<std::string> weightsfiles );

        float mvaValue(const pat::Electron& ele,
                const reco::Vertex& vertex,
                double rho,
                bool full5x5,
                bool printDebug = false);
    private:
        EGammaMvaEleEstimator *estimator_;
        EGammaMvaEleEstimatorCSA14 *estimatorCSA14_;
};
}
#endif
