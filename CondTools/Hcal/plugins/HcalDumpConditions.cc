
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
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c) override;

    template<class S, class SRcd> void dumpIt(S* myS, SRcd* mySRcd, const edm::Event& e, const edm::EventSetup& context, std::string name, const HcalTopology * topo);
    template<class S, class SRcd> void dumpIt(S* myS, SRcd* mySRcd, const edm::Event& e, const edm::EventSetup& context, std::string name);
    template<class S> void writeToFile(S* myS, const edm::Event& e, std::string name);

  private:
    std::string front;
    std::vector<std::string> mDumpRequest;
  };
  

  template<class S, class SRcd>
  void HcalDumpConditions::dumpIt(S* myS, SRcd* mySRcd, const edm::Event& e, const edm::EventSetup& context, std::string name, const HcalTopology * topo)
  {
    edm::ESHandle<S> p;
    if( name == "ChannelQuality") context.get<SRcd>().get("withTopo", p);
    else context.get<SRcd>().get(p);
    S* myobject = new S(*p.product());
    if( topo ) myobject->setTopo(topo);
    
    writeToFile(myobject, e, name);
    
    if ( context.get<SRcd>().validityInterval().first() == edm::IOVSyncValue::invalidIOVSyncValue() )
      std::cout << "error: invalid IOV sync value !" << std::endl;

  }


  template<class S, class SRcd>
  void HcalDumpConditions::dumpIt(S* myS, SRcd* mySRcd, const edm::Event& e, const edm::EventSetup& context, std::string name)
  {
    edm::ESHandle<S> p;
    context.get<SRcd>().get(p);
    S* myobject = new S(*p.product());

    writeToFile(myobject, e, name);
    
    if ( context.get<SRcd>().validityInterval().first() == edm::IOVSyncValue::invalidIOVSyncValue() )
      std::cout << "error: invalid IOV sync value !" << std::endl;

  }

  template<class S> void HcalDumpConditions::writeToFile(S* myS, const edm::Event& e, std::string name){
    int myrun = e.id().run();
    std::ostringstream file;
    file << front << name.c_str() << "_Run" << myrun << ".txt";
    std::ofstream outStream(file.str().c_str() );
    std::cout << "HcalDumpConditions: ---- Dumping " << name.c_str() << " ----" << std::endl;
    HcalDbASCIIIO::dumpObject (outStream, (*myS) );
  }

  void
   HcalDumpConditions::analyze(const edm::Event& e, const edm::EventSetup& context)
  {
    edm::ESHandle<HcalTopology> topology ;
    context.get<HcalRecNumberingRecord>().get( topology );
    const HcalTopology* topo=&(*topology);

    edm::ESHandle<HcalDbService> pSetup;
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("CalibrationsSet")) != mDumpRequest.end()
     || std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("CalibrationWidthsSet")) != mDumpRequest.end())
    {
      context.get<HcalDbRecord>().get( pSetup );
    }

    using namespace edm::eventsetup;
    std::cout <<"HcalDumpConditions::analyze-> I AM IN RUN NUMBER "<<e.id().run() <<std::endl;

    if (mDumpRequest.empty()) return;
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ElectronicsMap")) != mDumpRequest.end())
      dumpIt(new HcalElectronicsMap, new HcalElectronicsMapRcd, e,context,"ElectronicsMap");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("FrontEndMap")) != mDumpRequest.end())
      dumpIt(new HcalFrontEndMap, new HcalFrontEndMapRcd, e,context,"FrontEndMap");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("QIEData")) != mDumpRequest.end())
      dumpIt(new HcalQIEData(&(*topology)), new HcalQIEDataRcd, e,context,"QIEData", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("QIETypes")) != mDumpRequest.end())
      dumpIt(new HcalQIETypes(&(*topology)), new HcalQIETypesRcd, e,context,"QIETypes", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Pedestals")) != mDumpRequest.end())
      dumpIt(new HcalPedestals(&(*topology)), new HcalPedestalsRcd, e,context,"Pedestals", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("PedestalWidths")) != mDumpRequest.end())
      dumpIt(new HcalPedestalWidths(&(*topology)), new HcalPedestalWidthsRcd, e,context,"PedestalWidths", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Gains")) != mDumpRequest.end())
      dumpIt(new HcalGains(&(*topology)), new HcalGainsRcd, e,context,"Gains", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("GainWidths")) != mDumpRequest.end())
      dumpIt(new HcalGainWidths(&(*topology)), new HcalGainWidthsRcd, e,context,"GainWidths", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ChannelQuality")) != mDumpRequest.end())
      dumpIt(new HcalChannelQuality(&(*topology)), new HcalChannelQualityRcd, e,context,"ChannelQuality", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("RespCorrs")) != mDumpRequest.end())
      dumpIt(new HcalRespCorrs(&(*topology)), new HcalRespCorrsRcd, e,context,"RespCorrs", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ZSThresholds")) != mDumpRequest.end())
      dumpIt(new HcalZSThresholds(&(*topology)), new HcalZSThresholdsRcd, e,context,"ZSThresholds", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("L1TriggerObjects")) != mDumpRequest.end())
      dumpIt(new HcalL1TriggerObjects(&(*topology)), new HcalL1TriggerObjectsRcd, e,context,"L1TriggerObjects", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("TimeCorrs")) != mDumpRequest.end())
      dumpIt(new HcalTimeCorrs(&(*topology)), new HcalTimeCorrsRcd, e,context,"TimeCorrs", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("LUTCorrs")) != mDumpRequest.end())
      dumpIt(new HcalLUTCorrs(&(*topology)), new HcalLUTCorrsRcd, e,context,"LUTCorrs", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("PFCorrs")) != mDumpRequest.end())
      dumpIt(new HcalPFCorrs(&(*topology)), new HcalPFCorrsRcd, e,context,"PFCorrs", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ValidationCorrs")) != mDumpRequest.end())
      dumpIt(new HcalValidationCorrs(&(*topology)), new HcalValidationCorrsRcd, e,context,"ValidationCorrs", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("LutMetadata")) != mDumpRequest.end())
      dumpIt(new HcalLutMetadata(&(*topology)), new HcalLutMetadataRcd, e,context,"LutMetadata", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("DcsValues")) != mDumpRequest.end())
      dumpIt(new HcalDcsValues, new HcalDcsRcd, e,context,"DcsValues");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("DcsMap")) != mDumpRequest.end())
      dumpIt(new HcalDcsMap, new HcalDcsMapRcd, e,context,"DcsMap");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("RecoParams")) != mDumpRequest.end())
      dumpIt(new HcalRecoParams(&(*topology)), new HcalRecoParamsRcd, e,context,"RecoParams", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("TimingParams")) != mDumpRequest.end())
      dumpIt(new HcalTimingParams(&(*topology)), new HcalTimingParamsRcd, e,context,"TimingParams", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("LongRecoParams")) != mDumpRequest.end())
      dumpIt(new HcalLongRecoParams(&(*topology)), new HcalLongRecoParamsRcd, e,context,"LongRecoParams", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ZDCLowGainFractions")) != mDumpRequest.end())
      dumpIt(new HcalZDCLowGainFractions(&(*topology)), new HcalZDCLowGainFractionsRcd, e,context,"ZDCLowGainFractions", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("MCParams")) != mDumpRequest.end())
      dumpIt(new HcalMCParams(&(*topology)), new HcalMCParamsRcd, e,context,"MCParams", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("FlagHFDigiTimeParams")) != mDumpRequest.end())
      dumpIt(new HcalFlagHFDigiTimeParams(&(*topology)), new HcalFlagHFDigiTimeParamsRcd, e,context,"FlagHFDigiTimeParams", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("SiPMParameters")) != mDumpRequest.end())
      dumpIt(new HcalSiPMParameters(&(*topology)), new HcalSiPMParametersRcd, e,context,"SiPMParameters", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("SiPMCharacteristics")) != mDumpRequest.end())
      dumpIt(new HcalSiPMCharacteristics, new HcalSiPMCharacteristicsRcd, e,context,"SiPMCharacteristics");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("TPChannelParameters")) != mDumpRequest.end())
      dumpIt(new HcalTPChannelParameters(&(*topology)), new HcalTPChannelParametersRcd, e,context,"TPChannelParameters", topo);
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("TPParameters")) != mDumpRequest.end())
      dumpIt(new HcalTPParameters, new HcalTPParametersRcd, e,context,"TPParameters");
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("CalibrationsSet")) != mDumpRequest.end()){
      const HcalCalibrationsSet* tmp = pSetup->getHcalCalibrationsSet();
      writeToFile(tmp,e,"CalibrationsSet");
    }
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("CalibrationWidthsSet")) != mDumpRequest.end()){
      const HcalCalibrationWidthsSet* tmp = pSetup->getHcalCalibrationWidthsSet();
      writeToFile(tmp,e,"CalibrationWidthsSet");
    }

  }
  DEFINE_FWK_MODULE(HcalDumpConditions);
}
