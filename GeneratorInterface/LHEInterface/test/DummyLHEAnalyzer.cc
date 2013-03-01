#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <iostream>
using namespace std;
using namespace edm;
using namespace lhef;

class DummyLHEAnalyzer : public EDAnalyzer {
private: 
  bool dumpLHE_;
  bool checkPDG_;
public:
  explicit DummyLHEAnalyzer( const ParameterSet & cfg ) : 
    src_( cfg.getParameter<InputTag>( "src" ) )
  {
  }
private:
  void analyze( const Event & iEvent, const EventSetup & iSetup ) override {

    Handle<LHEEventProduct> evt;
    iEvent.getByLabel( src_, evt );

    const lhef::HEPEUP hepeup_ = evt->hepeup();

    const int nup_ = hepeup_.NUP; 
    const std::vector<int> idup_ = hepeup_.IDUP;
    const std::vector<lhef::HEPEUP::FiveVector> pup_ = hepeup_.PUP;

    std::cout << "Number of particles = " << nup_ << std::endl;

    if ( evt->pdf() != NULL ) {
      std::cout << "PDF scale = " << std::setw(14) << std::fixed << evt->pdf()->scalePDF << std::endl;  
      std::cout << "PDF 1 : id = " << std::setw(14) << std::fixed << evt->pdf()->id.first 
                << " x = " << std::setw(14) << std::fixed << evt->pdf()->x.first 
                << " xPDF = " << std::setw(14) << std::fixed << evt->pdf()->xPDF.first << std::endl;  
      std::cout << "PDF 2 : id = " << std::setw(14) << std::fixed << evt->pdf()->id.second 
                << " x = " << std::setw(14) << std::fixed << evt->pdf()->x.second 
                << " xPDF = " << std::setw(14) << std::fixed << evt->pdf()->xPDF.second << std::endl;  
    }

    for ( unsigned int icount = 0 ; icount < (unsigned int)nup_; icount++ ) {

      std::cout << "# " << std::setw(14) << std::fixed << icount 
                << std::setw(14) << std::fixed << idup_[icount] 
                << std::setw(14) << std::fixed << (pup_[icount])[0] 
                << std::setw(14) << std::fixed << (pup_[icount])[1] 
                << std::setw(14) << std::fixed << (pup_[icount])[2] 
                << std::setw(14) << std::fixed << (pup_[icount])[3] 
                << std::setw(14) << std::fixed << (pup_[icount])[4] 
                << std::endl;
    }


  }

  void beginRun(edm::Run const& iRun, edm::EventSetup const& es) override {


    Handle<LHERunInfoProduct> run;
    iRun.getByLabel( src_, run );
    
    const lhef::HEPRUP thisHeprup_ = run->heprup();

    std::cout << "HEPRUP \n" << std::endl;
    std::cout << "IDBMUP " << std::setw(14) << std::fixed << thisHeprup_.IDBMUP.first 
              << std::setw(14) << std::fixed << thisHeprup_.IDBMUP.second << std::endl; 
    std::cout << "EBMUP  " << std::setw(14) << std::fixed << thisHeprup_.EBMUP.first 
              << std::setw(14) << std::fixed << thisHeprup_.EBMUP.second << std::endl; 
    std::cout << "PDFGUP " << std::setw(14) << std::fixed << thisHeprup_.PDFGUP.first 
              << std::setw(14) << std::fixed << thisHeprup_.PDFGUP.second << std::endl; 
    std::cout << "PDFSUP " << std::setw(14) << std::fixed << thisHeprup_.PDFSUP.first 
              << std::setw(14) << std::fixed << thisHeprup_.PDFSUP.second << std::endl; 
    std::cout << "IDWTUP " << std::setw(14) << std::fixed << thisHeprup_.IDWTUP << std::endl; 
    std::cout << "NPRUP  " << std::setw(14) << std::fixed << thisHeprup_.NPRUP << std::endl; 
    std::cout << "        XSECUP " << std::setw(14) << std::fixed 
              << "        XERRUP " << std::setw(14) << std::fixed 
              << "        XMAXUP " << std::setw(14) << std::fixed 
              << "        LPRUP  " << std::setw(14) << std::fixed << std::endl;
    for ( unsigned int iSize = 0 ; iSize < thisHeprup_.XSECUP.size() ; iSize++ ) {
      std::cout  << std::setw(14) << std::fixed << thisHeprup_.XSECUP[iSize]
                 << std::setw(14) << std::fixed << thisHeprup_.XERRUP[iSize]
                 << std::setw(14) << std::fixed << thisHeprup_.XMAXUP[iSize]
                 << std::setw(14) << std::fixed << thisHeprup_.LPRUP[iSize] 
                 << std::endl;
    }
    std::cout << " " << std::endl;

  }

  InputTag src_;
};

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( DummyLHEAnalyzer );


