
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include <random>
#include <sys/wait.h>

namespace gen {

const std::vector<std::string> BaseHadronizer::theSharedResources;

  BaseHadronizer::BaseHadronizer( edm::ParameterSet const& ps ) :
  randomIndex_(-1),
  gridpackPaths_(1)
  {
    
    if (ps.exists("RandomizedParameters")) {
      std::vector<edm::ParameterSet> randomizedParameters = ps.getParameter<std::vector<edm::ParameterSet> >("RandomizedParameters");
      randomInitWeights_.resize(randomizedParameters.size());
      randomInitConfigDescriptions_.resize(randomizedParameters.size());
      gridpackPaths_.resize(randomizedParameters.size());
      for (unsigned int irand = 0; irand<randomizedParameters.size(); ++irand) {
        randomInitWeights_[irand] = randomizedParameters[irand].getParameter<double>("ConfigWeight");
        if (randomizedParameters[irand].exists("ConfigDescription")) {
          randomInitConfigDescriptions_[irand] = randomizedParameters[irand].getParameter<std::string>("ConfigDescription");
        }
        if (randomizedParameters[irand].exists("GridpackPath")) {
          gridpackPaths_[irand] = randomizedParameters[irand].getParameter<std::string>("GridpackPath");
        }
      }
    }
    else {
      if (ps.exists("GridpackPath")) {
        gridpackPaths_[0] = ps.getParameter<std::string>("GridpackPath");
      }
    }
    
    runInfo().setFilterEfficiency(
        ps.getUntrackedParameter<double>("filterEfficiency", -1.) );
    runInfo().setExternalXSecLO(
        GenRunInfoProduct::XSec(ps.getUntrackedParameter<double>("crossSection", -1.)) );
    runInfo().setExternalXSecNLO(
        GenRunInfoProduct::XSec(ps.getUntrackedParameter<double>("crossSectionNLO", -1.)) );

  }
  
  GenLumiInfoHeader *BaseHadronizer::getGenLumiInfoHeader() const {
    
    GenLumiInfoHeader *genLumiInfoHeader = new GenLumiInfoHeader();
    
    //fill information on randomized configs for parameter scans
    genLumiInfoHeader->setRandomConfigIndex(randomIndex_);
    if (randomIndex_>=0) {
      genLumiInfoHeader->setConfigDescription(randomInitConfigDescription());      
    }
    
    return genLumiInfoHeader;
    
  }

  void BaseHadronizer::randomizeIndex(edm::LuminosityBlock const& lumi, CLHEP::HepRandomEngine* rengine) {
    if (randomInitWeights_.size()>0) {
      //randomly select from a list of provided configuration sets (for parameter scans)
      
      //seeds std 32-bit mersene twister with HepRandomEngine state plus run and lumi section numbers
      //(random engine state will be the same for every lumi section in a job)
      std::vector<long unsigned int> seeds = rengine->put();
      seeds.push_back(lumi.id().run());
      seeds.push_back(lumi.id().luminosityBlock());
      std::seed_seq seedseq(seeds.begin(),seeds.end());
      std::mt19937 randgen(seedseq);
      std::discrete_distribution<int> randdist(randomInitWeights_.begin(),randomInitWeights_.end());
      
      randomIndex_ = randdist(randgen);
    }
  }
  
  void BaseHadronizer::generateLHE(edm::LuminosityBlock const& lumi, CLHEP::HepRandomEngine* rengine) {
        
    if (gridpackPath().empty()) {
      return;
    }
    
    //get random seed from HepRandomEngine state plus run and lumi section numbers
    //(random engine state will be the same for every lumi section in a job)
    std::vector<long unsigned int> seeds = rengine->put();
    seeds.push_back(lumi.id().run());
    seeds.push_back(lumi.id().luminosityBlock());
    std::seed_seq seedseq(seeds.begin(),seeds.end());
    std::array<unsigned int,1> lheseed;
    seedseq.generate(lheseed.begin(),lheseed.end());
    
    constexpr unsigned int maxseed = 30081*30081; //madgraph cannot handle seeds larger than this
    unsigned int seedval = lheseed[0]%(maxseed+1);

    unsigned int nevents = edm::pset::Registry::instance()->getMapped(lumi.processHistory().rbegin()->parameterSetID())->getParameter<edm::ParameterSet>("@main_input").getUntrackedParameter<unsigned int>("numberEventsInLuminosityBlock");
    
    std::ostringstream nevStream;
    nevStream << nevents;
    
    std::ostringstream randomStream;
    randomStream << seedval;
   
    edm::FileInPath script("GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh");
    const char *outfilename = "cmsgrid_final.lhe";
    
    char *args[5];
    args[0] = strdup(script.fullPath().c_str());
    args[1] = strdup(gridpackPath().c_str());
    args[2] = strdup(nevStream.str().c_str());
    args[3] = strdup(randomStream.str().c_str());
    args[4] = NULL;
    
    pid_t pid = fork();

    if (pid == -1) {
        // error, failed to fork()
      throw cms::Exception("BaseHadronizer::generateLHE") << "Unable to fork a child";
    } 
    else if (pid==0) {
      //child
      execvp(args[0],args);
      _exit(1);   // exec never returns
    }
    else {
      //parent
      int status;
      waitpid(pid, &status, 0);
      if (status) {
        throw cms::Exception("BaseHadronizer::generateLHE") << "Failed to execute script";
      }
    }
    FILE* lhef = std::fopen(outfilename, "r");
    if (!lhef) {
      throw cms::Exception("BaseHadronizer::generateLHE") << "Output file " << outfilename << " not found.";
    }
    std::fclose(lhef);

    lheFile_ = outfilename;
    
    for (int iarg=0; iarg<4; ++iarg) {
      delete[] args[iarg];
    }
    
  }
  
  void BaseHadronizer::cleanLHE() {
    if (lheFile_.empty()) {
      return;
    }
    
    std::remove(lheFile_.c_str());
  }

}
