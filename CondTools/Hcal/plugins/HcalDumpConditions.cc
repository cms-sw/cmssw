
/*----------------------------------------------------------------------

R.Ofierzynski - 2.Oct. 2007
   modified to dump all pedestals on screen, see 
   testHcalDBFake.cfg
   testHcalDBFrontier.cfg

July 29, 2009
   Gena Kukartsev
   Added HcalValidationCorrs
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

#include "CondFormats/DataRecord/interface/HcalAllRcds.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "CondFormats/HcalObjects/interface/AllObjects.h"

namespace edmtest
{
  class HcalDumpConditions : public edm::EDAnalyzer
  {
  public:
    explicit  HcalDumpConditions(edm::ParameterSet const& p) 
    {
      front = p.getUntrackedParameter<std::string>("outFilePrefix","Dump");
      mDumpRequest = p.getUntrackedParameter <std::vector <std::string> > ("dump", std::vector<std::string>());
    }

    explicit  HcalDumpConditions(int i) 
    { }
    virtual ~ HcalDumpConditions() { }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

    template<class S, class SRcd> void dumpIt(S* myS, SRcd* mySRcd, const edm::Event& e, const edm::EventSetup& context, std::string name);

  private:
    std::string front;
    std::vector<std::string> mDumpRequest;
  };
  

  template<class S, class SRcd>
  void HcalDumpConditions::dumpIt(S* myS, SRcd* mySRcd, const edm::Event& e, const edm::EventSetup& context, std::string name)
  {
    int myrun = e.id().run();
    edm::ESHandle<S> p;
    context.get<SRcd>().get(p);
    S* myobject = new S(*p.product());
    
    std::ostringstream file;
    file << front << name.c_str() << "_Run" << myrun << ".txt";
    std::ofstream outStream(file.str().c_str() );
    std::cout << "HcalDumpConditions: ---- Dumping " << name.c_str() << " ----" << std::endl;
    HcalDbASCIIIO::dumpObject (outStream, (*myobject) );

    if ( context.get<HcalPedestalsRcd>().validityInterval().first() == edm::IOVSyncValue::invalidIOVSyncValue() )
      std::cout << "error: invalid IOV sync value !" << std::endl;

  }


  void
   HcalDumpConditions::analyze(const edm::Event& e, const edm::EventSetup& context)
  {
    using namespace edm::eventsetup;
    std::cout <<"HcalDumpConditions::analyze-> I AM IN RUN NUMBER "<<e.id().run() <<std::endl;

    if (mDumpRequest.empty()) return;

    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ElectronicsMap")) != mDumpRequest.end())
      dumpIt(new HcalElectronicsMap, new HcalElectronicsMapRcd, e,context,"ElectronicsMap");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("QIEData")) != mDumpRequest.end())
      dumpIt(new HcalQIEData, new HcalQIEDataRcd, e,context,"QIEData");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Pedestals")) != mDumpRequest.end())
      dumpIt(new HcalPedestals(false), new HcalPedestalsRcd, e,context,"Pedestals");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("PedestalWidths")) != mDumpRequest.end())
      dumpIt(new HcalPedestalWidths(false), new HcalPedestalWidthsRcd, e,context,"PedestalWidths");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Gains")) != mDumpRequest.end())
      dumpIt(new HcalGains, new HcalGainsRcd, e,context,"Gains");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("GainWidths")) != mDumpRequest.end())
      dumpIt(new HcalGainWidths, new HcalGainWidthsRcd, e,context,"GainWidths");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ChannelQuality")) != mDumpRequest.end())
      dumpIt(new HcalChannelQuality, new HcalChannelQualityRcd, e,context,"ChannelQuality");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("RespCorrs")) != mDumpRequest.end())
      dumpIt(new HcalRespCorrs, new HcalRespCorrsRcd, e,context,"RespCorrs");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ZSThresholds")) != mDumpRequest.end())
      dumpIt(new HcalZSThresholds, new HcalZSThresholdsRcd, e,context,"ZSThresholds");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("L1TriggerObjects")) != mDumpRequest.end())
      dumpIt(new HcalL1TriggerObjects, new HcalL1TriggerObjectsRcd, e,context,"L1TriggerObjects");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("TimeCorrs")) != mDumpRequest.end())
      dumpIt(new HcalTimeCorrs, new HcalTimeCorrsRcd, e,context,"TimeCorrs");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("LUTCorrs")) != mDumpRequest.end())
      dumpIt(new HcalLUTCorrs, new HcalLUTCorrsRcd, e,context,"LUTCorrs");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("PFCorrs")) != mDumpRequest.end())
      dumpIt(new HcalPFCorrs, new HcalPFCorrsRcd, e,context,"PFCorrs");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ValidationCorrs")) != mDumpRequest.end())
      dumpIt(new HcalValidationCorrs, new HcalValidationCorrsRcd, e,context,"ValidationCorrs");
    
  }
  DEFINE_FWK_MODULE(HcalDumpConditions);
}
