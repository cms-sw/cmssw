#ifndef L1Trigger_Phase2L1ParticleFlow_corrector_h
#define L1Trigger_Phase2L1ParticleFlow_corrector_h
#include <TGraph.h>
#include <TH1.h>
#include <string>
#include <vector>

namespace l1t { class PFCluster; }

namespace l1tpf {
    class corrector { 
        public:
            corrector() : is2d_(false), neta_(0), nemf_(0), emfMax_(-1) {}
            corrector(const std::string &iFile, float emfMax=-1, bool debug=false);
            ~corrector();

            // no copy, but can move
            corrector(const corrector & corr) = delete;
            corrector & operator=(const corrector & corr) = delete;
            corrector(corrector && corr);
            corrector & operator=(corrector && corr);

            float correctedPt(float et, float emEt, float eta);
            void correctPt(l1t::PFCluster & cluster, float preserveEmEt=true);

            bool valid() const { return (index_.get() != nullptr); }

        private:
            std::unique_ptr<TH1> index_; 
            std::vector<TGraph*> corrections_;
            bool is2d_;
            unsigned int neta_, nemf_;
            float emfMax_;

            void init_(const std::string &iFile, bool debug) ;
    };
}
#endif
