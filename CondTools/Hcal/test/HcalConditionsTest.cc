
/*----------------------------------------------------------------------

R.Ofierzynski - 2.Oct. 2007
   modified to dump all pedestals on screen, see 
   testHcalDBFake.cfg
   testHcalDBFrontier.cfg

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/HcalZSThresholdsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"

namespace edmtest
{
  class HcalConditionsTest : public edm::EDAnalyzer
  {
  public:
    explicit  HcalConditionsTest(edm::ParameterSet const& p) 
    {
      front = p.getUntrackedParameter<std::string>("outFilePrefix","Dump");
    }

    explicit  HcalConditionsTest(int i) 
    { }
    virtual ~ HcalConditionsTest() { }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

    template<class S, class SRcd> void dumpIt(S* myS, SRcd* mySRcd, const edm::Event& e, const edm::EventSetup& context, std::string name);

  private:
    std::string front;
  };
  

  template<class S, class SRcd>
  void HcalConditionsTest::dumpIt(S* myS, SRcd* mySRcd, const edm::Event& e, const edm::EventSetup& context, std::string name)
  {
    int myrun = e.id().run();
    edm::ESHandle<S> p;
    context.get<SRcd>().get(p);
    S* myobject = new S(*p.product());
    
    std::ostringstream file;
    file << front << name.c_str() << "_Run" << myrun << ".txt";
    std::ofstream outStream(file.str().c_str() );
    std::cout << "HcalConditionsTest: ---- Dumping " << name.c_str() << " ----" << std::endl;
    HcalDbASCIIIO::dumpObject (outStream, (*myobject) );

    if ( context.get<HcalPedestalsRcd>().validityInterval().first() == edm::IOVSyncValue::invalidIOVSyncValue() )
      std::cout << "error: invalid IOV sync value !" << std::endl;

  }


  void
   HcalConditionsTest::analyze(const edm::Event& e, const edm::EventSetup& context)
  {
    using namespace edm::eventsetup;
    std::cout <<"HcalConditionsTest::analyze-> I AM IN RUN NUMBER "<<e.id().run() <<std::endl;

    dumpIt(new HcalElectronicsMap, new HcalElectronicsMapRcd, e,context,"ElectronicsMap");
    dumpIt(new HcalQIEData, new HcalQIEDataRcd, e,context,"QIEData");
    dumpIt(new HcalPedestals(false), new HcalPedestalsRcd, e,context,"Pedestals");
    dumpIt(new HcalPedestalWidths(false), new HcalPedestalWidthsRcd, e,context,"PedestalWidths");
    dumpIt(new HcalGains, new HcalGainsRcd, e,context,"Gains");
    dumpIt(new HcalGainWidths, new HcalGainWidthsRcd, e,context,"GainWidths");
    dumpIt(new HcalRespCorrs, new HcalRespCorrsRcd, e,context,"RespCorrs");
    dumpIt(new HcalChannelQuality, new HcalChannelQualityRcd, e,context,"ChannelQuality");
    dumpIt(new HcalZSThresholds, new HcalZSThresholdsRcd, e,context,"ZSThresholds");


  // get conditions
    edm::ESHandle<HcalDbService> conditions;
    context.get<HcalDbRecord>().get(conditions);

    int cell = HcalDetId (HcalBarrel, -1, 4, 1).rawId();
    
    const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);



//    int iov = 0;
//    // e-map
//    edm::ESHandle<HcalElectronicsMap> p;
//    context.get<HcalElectronicsMapRcd>().get(p);
//    HcalElectronicsMap* myemap = new HcalElectronicsMap(*p.product());
//    myemap->sort();
//
//    // dump emap
//    std::ostringstream filenameE;
//    filenameE << front << "HcalElectronicsMap" << "_" << iov << ".txt";
//    std::ofstream outStreamE(filenameE.str().c_str());
//    std::cout << "--- Dumping Electronics Map ---" << std::endl;
//    HcalDbASCIIIO::dumpObject (outStreamE, (*myemap) );
//
//    std::cout <<" Hcal peds for channel HB eta=15, phi=5, depth=2 "<<std::endl;
//    int channelID = HcalDetId (HcalBarrel, 15, 5, 2).rawId();
//    const HcalPedestals* myped=pPeds.product();
//    const HcalPedestalWidths* mypedW=pPedWs.product();
//    const HcalGains* mygain=pGains.product();
//    const HcalGainWidths* mygainW=pGainWs.product();
//
//    const float* values = myped->getValues (channelID);
//    if (values) std::cout << "pedestals for channel " << channelID << ": "
//			  << values [0] << '/' << values [1] << '/' << values [2] << '/' << values [3] << std::endl; 
//    values = mypedW->getValues (channelID);
//    if (values) std::cout << "pedestal widths for channel " << channelID << ": "
//			  << values [0] << '/' << values [1] << '/' << values [2] << '/' << values [3] << std::endl; 
//    values = mygain->getValues (channelID);
//    if (values) std::cout << "gains for channel " << channelID << ": "
//			  << values [0] << '/' << values [1] << '/' << values [2] << '/' << values [3] << std::endl; 
//    values = mygainW->getValues (channelID);
//    if (values) std::cout << "gain widts for channel " << channelID << ": "
//			  << values [0] << '/' << values [1] << '/' << values [2] << '/' << values [3] << std::endl; 
  }
  DEFINE_FWK_MODULE(HcalConditionsTest);
}
