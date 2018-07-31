
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
    ~ HcalDumpConditions() override { }
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

    template<class S, class SRcd> void dumpIt(const std::vector<std::string>& mDumpRequest,
                                              const edm::Event& e,
                                              const edm::EventSetup& context,
                                              const std::string name,
                                              const HcalTopology * topo,
                                              const std::string label="");
    template<class S, class SRcd> void dumpIt(const std::vector<std::string>& mDumpRequest,
                                              const edm::Event& e,
                                              const edm::EventSetup& context,
                                              const std::string name);
    template<class S> void writeToFile(const S& myS, const edm::Event& e, const std::string name);

  private:
    std::string front;
    std::vector<std::string> mDumpRequest;
  };
  

  template<class S, class SRcd>
  void HcalDumpConditions::dumpIt(const std::vector<std::string>& mDumpRequest,
                                  const edm::Event& e,
                                  const edm::EventSetup& context,
                                  const std::string name,
                                  const HcalTopology * topo,
                                  const std::string label)
  {
    edm::ESHandle<S> p;
    if(!label.empty()) context.get<SRcd>().get(label, p);
    else context.get<SRcd>().get(p);
    S myobject(*p.product());
    if( topo ) myobject.setTopo(topo);
    
    writeToFile(myobject, e, name);
    
    if ( context.get<SRcd>().validityInterval().first() == edm::IOVSyncValue::invalidIOVSyncValue() )
      std::cout << "error: invalid IOV sync value !" << std::endl;

  }


  template<class S, class SRcd>
  void HcalDumpConditions::dumpIt(const std::vector<std::string>& mDumpRequest,
                                  const edm::Event& e,
                                  const edm::EventSetup& context,
                                  const std::string name)
  {
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), name) != mDumpRequest.end())
    {
        edm::ESHandle<S> p;
        context.get<SRcd>().get(p);
        S myobject(*p.product());

        writeToFile(myobject, e, name);
        
        if ( context.get<SRcd>().validityInterval().first() == edm::IOVSyncValue::invalidIOVSyncValue() )
          std::cout << "error: invalid IOV sync value !" << std::endl;
    }

  }

  template<class S> void HcalDumpConditions::writeToFile(const S& myS, const edm::Event& e, const std::string name){
    int myrun = e.id().run();
    std::ostringstream file;
    file << front << name.c_str() << "_Run" << myrun << ".txt";
    std::ofstream outStream(file.str().c_str() );
    std::cout << "HcalDumpConditions: ---- Dumping " << name << " ----" << std::endl;
    HcalDbASCIIIO::dumpObject (outStream, myS );
  }

  void
   HcalDumpConditions::analyze(const edm::Event& e, const edm::EventSetup& context)
  {
    edm::ESHandle<HcalTopology> topology ;
    context.get<HcalRecNumberingRecord>().get( topology );
    const HcalTopology* topo=&(*topology);

    using namespace edm::eventsetup;
    std::cout <<"HcalDumpConditions::analyze-> I AM IN RUN NUMBER "<<e.id().run() <<std::endl;

    if (mDumpRequest.empty()) return;

    // dumpIt called for all possible ValueMaps. The function checks if the dump is actually requested.
    dumpIt<HcalElectronicsMap,       HcalElectronicsMapRcd>      (mDumpRequest, e, context,"ElectronicsMap"               );
    dumpIt<HcalFrontEndMap,          HcalFrontEndMapRcd>         (mDumpRequest, e, context,"FrontEndMap"                  );
    dumpIt<HcalQIEData,              HcalQIEDataRcd>             (mDumpRequest, e, context,"QIEData",                 topo);
    dumpIt<HcalQIETypes,             HcalQIETypesRcd>            (mDumpRequest, e, context,"QIETypes",                topo);
    dumpIt<HcalPedestals,            HcalPedestalsRcd>           (mDumpRequest, e, context,"Pedestals",               topo);
    dumpIt<HcalPedestalWidths,       HcalPedestalWidthsRcd>      (mDumpRequest, e, context,"PedestalWidths",          topo);
    dumpIt<HcalPedestals,            HcalPedestalsRcd>           (mDumpRequest, e, context,"EffectivePedestals",      topo, "effective");
    dumpIt<HcalPedestalWidths,       HcalPedestalWidthsRcd>      (mDumpRequest, e, context,"EffectivePedestalWidths", topo, "effective");
    dumpIt<HcalGains,                HcalGainsRcd>               (mDumpRequest, e, context,"Gains",                   topo);
    dumpIt<HcalGainWidths,           HcalGainWidthsRcd>          (mDumpRequest, e, context,"GainWidths",              topo);
    dumpIt<HcalChannelQuality,       HcalChannelQualityRcd>      (mDumpRequest, e, context,"ChannelQuality",          topo, "withTopo" );
    dumpIt<HcalRespCorrs,            HcalRespCorrsRcd>           (mDumpRequest, e, context,"RespCorrs",               topo);
    dumpIt<HcalZSThresholds,         HcalZSThresholdsRcd>        (mDumpRequest, e, context,"ZSThresholds",            topo);
    dumpIt<HcalL1TriggerObjects,     HcalL1TriggerObjectsRcd>    (mDumpRequest, e, context,"L1TriggerObjects",        topo);
    dumpIt<HcalTimeCorrs,            HcalTimeCorrsRcd>           (mDumpRequest, e, context,"TimeCorrs",               topo);
    dumpIt<HcalLUTCorrs,             HcalLUTCorrsRcd>            (mDumpRequest, e, context,"LUTCorrs",                topo);
    dumpIt<HcalPFCorrs,              HcalPFCorrsRcd>             (mDumpRequest, e, context,"PFCorrs",                 topo);
    dumpIt<HcalValidationCorrs,      HcalValidationCorrsRcd>     (mDumpRequest, e, context,"ValidationCorrs",         topo);
    dumpIt<HcalLutMetadata,          HcalLutMetadataRcd>         (mDumpRequest, e, context,"LutMetadata",             topo);
    dumpIt<HcalDcsValues,            HcalDcsRcd>                 (mDumpRequest, e, context,"DcsValues"                    );
    dumpIt<HcalDcsMap,               HcalDcsMapRcd>              (mDumpRequest, e, context,"DcsMap"                       );
    dumpIt<HcalRecoParams,           HcalRecoParamsRcd>          (mDumpRequest, e, context,"RecoParams",              topo);
    dumpIt<HcalTimingParams,         HcalTimingParamsRcd>        (mDumpRequest, e, context,"TimingParams",            topo);
    dumpIt<HcalLongRecoParams,       HcalLongRecoParamsRcd>      (mDumpRequest, e, context,"LongRecoParams",          topo);
    dumpIt<HcalZDCLowGainFractions,  HcalZDCLowGainFractionsRcd> (mDumpRequest, e, context,"ZDCLowGainFractions",     topo);
    dumpIt<HcalMCParams,             HcalMCParamsRcd>            (mDumpRequest, e, context,"MCParams",                topo);
    dumpIt<HcalFlagHFDigiTimeParams, HcalFlagHFDigiTimeParamsRcd>(mDumpRequest, e, context,"FlagHFDigiTimeParams",    topo);
    dumpIt<HcalSiPMParameters,       HcalSiPMParametersRcd>      (mDumpRequest, e, context,"SiPMParameters",          topo);
    dumpIt<HcalSiPMCharacteristics,  HcalSiPMCharacteristicsRcd> (mDumpRequest, e, context,"SiPMCharacteristics"          );
    dumpIt<HcalTPChannelParameters,  HcalTPChannelParametersRcd> (mDumpRequest, e, context,"TPChannelParameters",     topo);
    dumpIt<HcalTPParameters,         HcalTPParametersRcd>        (mDumpRequest, e, context,"TPParameters"                 );

    edm::ESHandle<HcalDbService> pSetup;
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("CalibrationsSet")) != mDumpRequest.end()
     || std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("CalibrationWidthsSet")) != mDumpRequest.end())
    {
        context.get<HcalDbRecord>().get( pSetup );
        if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("CalibrationsSet")) != mDumpRequest.end()){
          writeToFile(*pSetup->getHcalCalibrationsSet(),e,"CalibrationsSet");
        }
        if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("CalibrationWidthsSet")) != mDumpRequest.end()){
          writeToFile(*pSetup->getHcalCalibrationWidthsSet(),e,"CalibrationWidthsSet");
        }
    }
  }
  DEFINE_FWK_MODULE(HcalDumpConditions);
}
