#ifndef MuEnrichRenormalizer_H
#define MuEnrichRenormalizer_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "TProfile.h"

// forward declarations
class TFile;
class TH1D;
class TH2D;
class TTree;

class MuEnrichRenormalizer : public edm::EDAnalyzer
{
  public:
     //
      explicit MuEnrichRenormalizer( const edm::ParameterSet& ) ;
      virtual ~MuEnrichRenormalizer() {} // no need to delete ROOT stuff
                                   // as it'll be deleted upon closing TFile
      
      virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
      virtual void beginJob() ;
      virtual void endJob() ;
 
      //     HepMC::GenEvent  *evt;
 

   private:
      int type,genLight,genBC, anaLight, anaBC;
      double genIntlumi, anaIntlumi,rwbc,rwlight;

};

#endif
