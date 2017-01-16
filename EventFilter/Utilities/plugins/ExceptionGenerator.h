#include "TH1D.h"

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/stream/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

//#include "EventFilter/Utilities/interface/ModuleWeb.h"

#include <vector>
#include <string>

namespace evf{
    class ExceptionGenerator : public edm::stream::EDAnalyzer<>
    {
    public:
      static const int menu_items = 14;
      static const std::string menu[menu_items];
						   
      explicit ExceptionGenerator( const edm::ParameterSet&);
      ~ExceptionGenerator(){};
      void beginRun(edm::Run& r, const edm::EventSetup& iSetup);
      void analyze(const edm::Event & e, const edm::EventSetup& c);
      void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

    private:
      int actionId_;
      unsigned int intqualifier_;
      double qualifier2_;
      std::string qualifier_;
      bool actionRequired_;
      std::string original_referrer_;
      TH1D* timingHisto_;
      timeval tv_start_;
    };
  }

