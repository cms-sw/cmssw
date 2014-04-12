#ifndef JetProducers_PileupJPTJetIdAlgo_h
#define JetProducers_PileupJPTJetIdAlgo_h

#include "DataFormats/JetReco/interface/JPTJet.h"

// user include files
#include <string>
#include <memory>
#include <map>
#include<fstream>
#include<iomanip>
#include<iostream>
#include<vector>

using namespace std; 
using namespace reco;

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSets;
}
// For MVA analysis

#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

namespace cms
{

class PileupJPTJetIdAlgo
{
public:  

  PileupJPTJetIdAlgo(const edm::ParameterSet& fParameters);

  virtual ~PileupJPTJetIdAlgo();

  void bookMVAReader(); 

  float fillJPTBlock(const reco::JPTJet* jet 
                  );
private:
     int verbosity;
// Variables for multivariate analysis

     float Nvtx,PtJ,EtaJ,Beta,MultCalo,dAxis1c,dAxis2c,MultTr,dAxis1t,dAxis2t;
     TMVA::Reader * reader_; 
     TMVA::Reader * readerF_;
     std::string    tmvaWeights_, tmvaWeightsF_, tmvaMethod_;
};
}
#endif
