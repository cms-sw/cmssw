#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/EcalAlgo/interface/WriteESAlignments.h"

typedef WriteESAlignments WEA ;

class TestWriteESAlignments : public edm::one::EDAnalyzer<>
{
public:

  explicit TestWriteESAlignments(const edm::ParameterSet& /*iConfig*/)
    : nEventCalls_(0) {}
  ~TestWriteESAlignments() {}
  
  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  
  unsigned int nEventCalls_;
};
  
void TestWriteESAlignments::analyze(const edm::Event& /*evt*/, const edm::EventSetup& evtSetup)
{
   if (nEventCalls_ > 0) {
     edm::LogInfo("TestWriteESAlignments") << "Writing to DB to be done only once, "
	       << "set 'untracked PSet maxEvents = {untracked int32 input = 1}'."
	       << "(Your writing should be fine.)";
     return;
   }


   static const unsigned int nA ( EcalPreshowerGeometry::numberOfAlignments() ) ;

   typedef std::vector<double> DVec ;

   DVec alphaVec   ( nA, 0 ) ;
   DVec betaVec    ( nA, 0 ) ;
   DVec gammaVec   ( nA, 0 ) ;
   DVec xtranslVec ( nA, 0 ) ;
   DVec ytranslVec ( nA, 0 ) ;
   DVec ztranslVec ( nA, 0 ) ;

   const WriteESAlignments wea( evtSetup   ,
				alphaVec   ,
				betaVec    ,
				gammaVec   ,
				xtranslVec ,
				ytranslVec ,
				ztranslVec  ) ;
   
   edm::LogInfo("TestWriteESAlignments") << "Done!";
   nEventCalls_++;
}

DEFINE_FWK_MODULE(TestWriteESAlignments);
