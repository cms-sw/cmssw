
/*----------------------------------------------------------------------

R.Ofierzynski - 2.Oct. 2007
   modified to dump all pedestals on screen, see 
   testHcalDBFake.cfg
   testHcalDBFrontier.cfg

July 29, 2009       Added HcalValidationCorrs - Gena Kukartsev
September 21, 2009  Added HcalLutMetadata - Gena Kukartsev
   
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
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

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

    if ( context.get<SRcd>().validityInterval().first() == edm::IOVSyncValue::invalidIOVSyncValue() )
      std::cout << "error: invalid IOV sync value !" << std::endl;

  }


  void
   HcalDumpConditions::analyze(const edm::Event& e, const edm::EventSetup& context)
  {
    edm::ESHandle<HcalTopology> topology ;
    context.get<HcalRecNumberingRecord>().get( topology );

    using namespace edm::eventsetup;
    std::cout <<"HcalDumpConditions::analyze-> I AM IN RUN NUMBER "<<e.id().run() <<std::endl;

    if (mDumpRequest.empty()) return;
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ElectronicsMap")) != mDumpRequest.end())
      dumpIt(new HcalElectronicsMap, new HcalElectronicsMapRcd, e,context,"ElectronicsMap");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("QIEData")) != mDumpRequest.end())
      dumpIt(new HcalQIEData(&(*topology)), new HcalQIEDataRcd, e,context,"QIEData");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Pedestals")) != mDumpRequest.end())
      dumpIt(new HcalPedestals(&(*topology)), new HcalPedestalsRcd, e,context,"Pedestals");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("PedestalWidths")) != mDumpRequest.end())
      dumpIt(new HcalPedestalWidths(&(*topology)), new HcalPedestalWidthsRcd, e,context,"PedestalWidths");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Gains")) != mDumpRequest.end())
      dumpIt(new HcalGains(&(*topology)), new HcalGainsRcd, e,context,"Gains");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("GainWidths")) != mDumpRequest.end())
      dumpIt(new HcalGainWidths(&(*topology)), new HcalGainWidthsRcd, e,context,"GainWidths");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ChannelQuality")) != mDumpRequest.end())
      dumpIt(new HcalChannelQuality(&(*topology)), new HcalChannelQualityRcd, e,context,"ChannelQuality");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("RespCorrs")) != mDumpRequest.end())
      dumpIt(new HcalRespCorrs(&(*topology)), new HcalRespCorrsRcd, e,context,"RespCorrs");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ZSThresholds")) != mDumpRequest.end())
      dumpIt(new HcalZSThresholds(&(*topology)), new HcalZSThresholdsRcd, e,context,"ZSThresholds");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("L1TriggerObjects")) != mDumpRequest.end())
      dumpIt(new HcalL1TriggerObjects(&(*topology)), new HcalL1TriggerObjectsRcd, e,context,"L1TriggerObjects");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("TimeCorrs")) != mDumpRequest.end())
      dumpIt(new HcalTimeCorrs(&(*topology)), new HcalTimeCorrsRcd, e,context,"TimeCorrs");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("LUTCorrs")) != mDumpRequest.end())
      dumpIt(new HcalLUTCorrs(&(*topology)), new HcalLUTCorrsRcd, e,context,"LUTCorrs");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("PFCorrs")) != mDumpRequest.end())
      dumpIt(new HcalPFCorrs(&(*topology)), new HcalPFCorrsRcd, e,context,"PFCorrs");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ValidationCorrs")) != mDumpRequest.end())
      dumpIt(new HcalValidationCorrs(&(*topology)), new HcalValidationCorrsRcd, e,context,"ValidationCorrs");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("LutMetadata")) != mDumpRequest.end())
      dumpIt(new HcalLutMetadata(&(*topology)), new HcalLutMetadataRcd, e,context,"LutMetadata");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("DcsValues")) != mDumpRequest.end())
      dumpIt(new HcalDcsValues, new HcalDcsRcd, e,context,"DcsValues");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("DcsMap")) != mDumpRequest.end())
      dumpIt(new HcalDcsMap, new HcalDcsMapRcd, e,context,"DcsMap");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("CholeskyMatrices")) != mDumpRequest.end())
      dumpIt(new HcalCholeskyMatrices(&(*topology)), new HcalCholeskyMatricesRcd, e,context,"CholeskyMatrices");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("RecoParams")) != mDumpRequest.end())
      dumpIt(new HcalRecoParams(&(*topology)), new HcalRecoParamsRcd, e,context,"RecoParams");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("TimingParams")) != mDumpRequest.end())
      dumpIt(new HcalTimingParams(&(*topology)), new HcalTimingParamsRcd, e,context,"TimingParams");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("LongRecoParams")) != mDumpRequest.end())
      dumpIt(new HcalLongRecoParams(&(*topology)), new HcalLongRecoParamsRcd, e,context,"LongRecoParams");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("MCParams")) != mDumpRequest.end())
      dumpIt(new HcalMCParams(&(*topology)), new HcalMCParamsRcd, e,context,"MCParams");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("FlagHFDigiTimeParams")) != mDumpRequest.end())
      dumpIt(new HcalFlagHFDigiTimeParams(&(*topology)), new HcalFlagHFDigiTimeParamsRcd, e,context,"FlagHFDigiTimeParams");

    
  }
  DEFINE_FWK_MODULE(HcalDumpConditions);
}
