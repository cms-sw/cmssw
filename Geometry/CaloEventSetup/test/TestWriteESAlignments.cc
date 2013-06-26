#include <string>
#include <map>
#include <vector>

// Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Utilities/General/interface/ClassName.h"
#include "DataFormats/DetId/interface/DetId.h"

// Database
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"

#include "Geometry/EcalAlgo/interface/WriteESAlignments.h"
typedef WriteESAlignments WEA ;

class TestWriteESAlignments : public edm::EDAnalyzer
{
public:

  explicit TestWriteESAlignments(const edm::ParameterSet& /*iConfig*/)
    : nEventCalls_(0) {}
  ~TestWriteESAlignments() {}
  
//  template<typename T>
//  void writeAlignments(const edm::EventSetup& evtSetup);

  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);

private:
  
  unsigned int nEventCalls_;
};
  
void TestWriteESAlignments::analyze(const edm::Event& /*evt*/, const edm::EventSetup& evtSetup)
{
   if (nEventCalls_ > 0) {
     std::cout << "Writing to DB to be done only once, "
	       << "set 'untracked PSet maxEvents = {untracked int32 input = 1}'."
	       << "(Your writing should be fine.)" << std::endl;
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
   
   std::cout << "done!" << std::endl;
   nEventCalls_++;
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestWriteESAlignments);
