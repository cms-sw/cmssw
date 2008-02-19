/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
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
    std::ofstream DBNoiseMatrixFile("dbmatrix.dat",std::ios::out);
    int counter=0;
    using namespace edm::eventsetup;
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::ESHandle<CSCDBNoiseMatrix> pNoiseMatrix;
    context.get<CSCDBNoiseMatrixRcd>().get(pNoiseMatrix);

    const CSCDBNoiseMatrix* myNoiseMatrix=pNoiseMatrix.product();
    int i;
    for( i=0; i<CSCDBNoiseMatrix::ArraySize; ++i ){
      counter++;
      DBNoiseMatrixFile<<counter<<"  "<<myNoiseMatrix->matrix[i].elem33<<" "<<myNoiseMatrix->matrix[i].elem34<<"  "<<myNoiseMatrix->matrix[i].elem44<<"  "<<myNoiseMatrix->matrix[i].elem35<<" "<<myNoiseMatrix->matrix[i].elem45<<"  "<<myNoiseMatrix->matrix[i].elem55<<"  "<<myNoiseMatrix->matrix[i].elem46<<"  "<<myNoiseMatrix->matrix[i].elem56<<"  "<<myNoiseMatrix->matrix[i].elem66<<"  "<<myNoiseMatrix->matrix[i].elem57<<"  "<<myNoiseMatrix->matrix[i].elem67<<"  "<<myNoiseMatrix->matrix[i].elem77<<std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCNoiseMatrixDBReadAnalyzer);
}

