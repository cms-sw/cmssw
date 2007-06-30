#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/JetReco/interface/Jet.h"

#include "RecoBTag/MCTools/interface/JetFlavourIdentifier.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include <TH1.h>
#include <string>

namespace reco { typedef std::vector<std::pair<edm::RefToBase<Jet>, float>  > JetChargeCollection; }

class JetChargeAnalyzer : public edm::EDAnalyzer {
    public:
        explicit JetChargeAnalyzer(const edm::ParameterSet&);
        ~JetChargeAnalyzer() {}

        virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
        virtual void beginJob(const edm::EventSetup& iSetup);
        virtual void endJob(const edm::EventSetup& iSetup);
    private:
        // physics stuff
        edm::InputTag         src_;
        double                minET_;
        JetFlavourIdentifier  jfi_;
        // plot stuff
        std::string           dir_;
        TH1D *charge_[12];
        
};

const int   pdgIds[12] = {  0 ,  1 ,  -1 ,  2 ,  -2 ,  3 ,  -3 ,  4 ,  -4 ,  5 ,  -5 , 21  };
const char* pdgs  [12] = { "?", "u", "-u", "d", "-d", "s", "-s", "c", "-c", "b", "-b", "g" };

JetChargeAnalyzer::JetChargeAnalyzer(const edm::ParameterSet &iConfig) :
        src_(iConfig.getParameter<edm::InputTag>("src")),
        minET_(iConfig.getParameter<double>("minET")),
        jfi_(iConfig.getParameter<edm::ParameterSet>("jetIdParameters")),
        dir_(iConfig.getParameter<std::string>("dir")) {
    edm::Service<TFileService> fs;
    TFileDirectory cwd = fs->mkdir(dir_.c_str());
    char buff[255],biff[255];
    for (int i = 0; i < 12; i++) {
        sprintf(biff,"jch_id_%s%d", ( pdgIds[i] >= 0 ? "p" : "m" ), abs(pdgIds[i]) );
        sprintf(buff,"Jet charge for '%s' jets (pdgId %d) [ET > %f]", pdgs[i], pdgIds[i],minET_);
        charge_[i] = cwd.make<TH1D>(biff,buff,22,-1.1,1.1);
    }
}

void JetChargeAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    using namespace edm; using namespace reco;
    
    Handle<JetChargeCollection> hJC;
    iEvent.getByLabel(src_, hJC);
    jfi_.readEvent(iEvent);
    for (JetChargeCollection::const_iterator it = hJC->begin(), ed = hJC->end(); it != ed; ++it) {
        const Jet &jet = *(it->first);
        if (jet.et() < minET_) continue;
        int id = jfi_.identifyBasedOnPartons(jet).mainFlavour();
        int k;
        for (k = 0; k < 12; k++) { if (id == pdgIds[k]) break; }
        if (k == 12) {
                std::cerr << "Error: jet with flavour " << id << ". !??" << std::endl;
                continue;
        }
        charge_[k]->Fill(it->second);
    }
    
}
void JetChargeAnalyzer::beginJob(const edm::EventSetup& iSetup) {
}
void JetChargeAnalyzer::endJob(const edm::EventSetup& iSetup) {
}

DEFINE_FWK_MODULE(JetChargeAnalyzer);
