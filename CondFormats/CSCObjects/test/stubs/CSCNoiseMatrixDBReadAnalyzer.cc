/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"

using namespace std;

namespace edmtest
{
  class CSCNoiseMatrixDBReadAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit  CSCNoiseMatrixDBReadAnalyzer(edm::ParameterSet const& p) 
    { }
    explicit  CSCNoiseMatrixDBReadAnalyzer(int i) 
    { }
    virtual ~ CSCNoiseMatrixDBReadAnalyzer() { }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
  };
  
  void
   CSCNoiseMatrixDBReadAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context)
  {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::ESHandle<CSCDBNoiseMatrix> pNoiseMatrix;
    context.get<CSCDBNoiseMatrixRcd>().get(pNoiseMatrix);

    const CSCDBNoiseMatrix* myNoiseMatrix=pNoiseMatrix.product();
    std::vector<CSCDBNoiseMatrix::Item>::const_iterator it;

    for( it=myNoiseMatrix->matrix.begin();it!=myNoiseMatrix->matrix.end(); ++it ){
      // no global variables
      //   counter++;
      // DBNoiseMatrixFile<<counter<<"  "<<it->elem33<<" "<<it->elem34<<"  "<<it->elem44<<"  "<<it->elem35<<" "<<it->elem45<<"  "<<it->elem55<<"  "<<it->elem46<<"  "<<it->elem56<<"  "<<it->elem66<<"  "<<it->elem57<<"  "<<it->elem67<<"  "<<it->elem77<<std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCNoiseMatrixDBReadAnalyzer);
}

