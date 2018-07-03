#ifndef HZZ4muAnalyzer_H
#define HZZ4muAnalyzer_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

// forward declarations
class TFile;
class TH1D;

class HZZ4muAnalyzer : public edm::EDAnalyzer
{

   public:

      //
      explicit HZZ4muAnalyzer( const edm::ParameterSet& ) ;
      virtual ~HZZ4muAnalyzer() {} // no need to delete ROOT stuff
                                   // as it'll be deleted upon closing TFile

      virtual void analyze( const edm::Event&, const edm::EventSetup&) override;
      virtual void beginJob() override ;
      virtual void endJob() override ;

   private:

     //
     edm::EDGetTokenT<edm::HepMCProduct> fToken ;
     std::string fOutputFileName ;
     TFile*      fOutputFile ;
     TH1D*       fHist2muMass ;
     TH1D*       fHist4muMass ;
     TH1D*       fHistZZMass ;

};

#endif
