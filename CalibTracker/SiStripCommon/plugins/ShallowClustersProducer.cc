#include "CalibTracker/SiStripCommon/interface/ShallowClustersProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"
#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "boost/foreach.hpp"

ShallowClustersProducer::ShallowClustersProducer(const edm::ParameterSet& iConfig) 
  : theClustersLabel(iConfig.getParameter<edm::InputTag>("Clusters")),
    Prefix(iConfig.getParameter<std::string>("Prefix") )
{
  produces <std::vector<unsigned> >    ( Prefix + "number"       );
  produces <std::vector<unsigned> >    ( Prefix + "width"        );
  produces <std::vector<float> >       ( Prefix + "variance"     );
  produces <std::vector<float> >       ( Prefix + "barystrip"    );
  produces <std::vector<float> >       ( Prefix + "middlestrip"  );
  produces <std::vector<unsigned> >    ( Prefix + "charge"       );
  produces <std::vector<float> >       ( Prefix + "noise"        );
  produces <std::vector<float> >       ( Prefix + "ston"         );
  produces <std::vector<unsigned> >    ( Prefix + "seedstrip"    );
  produces <std::vector<unsigned> >    ( Prefix + "seedindex"    );
  produces <std::vector<unsigned> >    ( Prefix + "seedcharge"   );
  produces <std::vector<float> >       ( Prefix + "seednoise"    );
  produces <std::vector<float> >       ( Prefix + "seedgain"     );
  produces <std::vector<unsigned> >    ( Prefix + "qualityisbad" );

  produces <std::vector<float> >       ( Prefix + "rawchargeC"   );
  produces <std::vector<float> >       ( Prefix + "rawchargeL"   );
  produces <std::vector<float> >       ( Prefix + "rawchargeR"   );
  produces <std::vector<float> >       ( Prefix + "rawchargeLL"   );
  produces <std::vector<float> >       ( Prefix + "rawchargeRR"   );
  produces <std::vector<float> >       ( Prefix + "eta"          );
  produces <std::vector<float> >       ( Prefix + "foldedeta"    );
  produces <std::vector<float> >       ( Prefix + "etaX"         );
  produces <std::vector<float> >       ( Prefix + "etaasymm"     );
  produces <std::vector<float> >       ( Prefix + "outsideasymm");
  produces <std::vector<float> >       ( Prefix + "neweta");
  produces <std::vector<float> >       ( Prefix + "newetaerr");
  
  produces <std::vector<unsigned> >    ( Prefix + "detid"         );
  produces <std::vector<int> >         ( Prefix + "subdetid"      );
  produces <std::vector<int> >         ( Prefix + "module"        );
  produces <std::vector<int> >         ( Prefix + "side"          );
  produces <std::vector<int> >         ( Prefix + "layerwheel"    );
  produces <std::vector<int> >         ( Prefix + "stringringrod" );
  produces <std::vector<int> >         ( Prefix + "petal"         );
  produces <std::vector<int> >         ( Prefix + "stereo"        );

}

void ShallowClustersProducer::
produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::auto_ptr<std::vector<unsigned> >       number       ( new std::vector<unsigned>(7,0) );
  std::auto_ptr<std::vector<unsigned> >       width        ( new std::vector<unsigned>() );
  std::auto_ptr<std::vector<float> >          variance     ( new std::vector<float>() );
  std::auto_ptr<std::vector<float> >          barystrip    ( new std::vector<float>() );
  std::auto_ptr<std::vector<float> >          middlestrip  ( new std::vector<float>() );
  std::auto_ptr<std::vector<unsigned> >       charge       ( new std::vector<unsigned>() );
  std::auto_ptr<std::vector<float> >          noise        ( new std::vector<float>() );
  std::auto_ptr<std::vector<float> >          ston         ( new std::vector<float>() );
  std::auto_ptr<std::vector<unsigned> >       seedstrip    ( new std::vector<unsigned>() );
  std::auto_ptr<std::vector<unsigned> >       seedindex    ( new std::vector<unsigned>() );
  std::auto_ptr<std::vector<unsigned> >       seedcharge   ( new std::vector<unsigned>() );
  std::auto_ptr<std::vector<float> >          seednoise    ( new std::vector<float>() );
  std::auto_ptr<std::vector<float> >          seedgain     ( new std::vector<float>() );
  std::auto_ptr<std::vector<unsigned> >       qualityisbad ( new std::vector<unsigned>() );

  std::auto_ptr<std::vector<float> >          rawchargeC   ( new std::vector<float>() );
  std::auto_ptr<std::vector<float> >          rawchargeL   ( new std::vector<float>() );
  std::auto_ptr<std::vector<float> >          rawchargeR   ( new std::vector<float>() );
  std::auto_ptr<std::vector<float> >          rawchargeLL  ( new std::vector<float>() );
  std::auto_ptr<std::vector<float> >          rawchargeRR  ( new std::vector<float>() );
  std::auto_ptr<std::vector<float> >          etaX         ( new std::vector<float>() );
  std::auto_ptr<std::vector<float> >          eta          ( new std::vector<float>() );
  std::auto_ptr<std::vector<float> >          foldedeta    ( new std::vector<float>() );
  std::auto_ptr<std::vector<float> >          etaasymm     ( new std::vector<float>() );
  std::auto_ptr<std::vector<float> >          outsideasymm ( new std::vector<float>() );
  std::auto_ptr<std::vector<float> >          neweta       ( new std::vector<float>() );
  std::auto_ptr<std::vector<float> >          newetaerr    ( new std::vector<float>() );
  
  std::auto_ptr<std::vector<unsigned> >       detid          ( new std::vector<unsigned>() );
  std::auto_ptr<std::vector<int> >            subdetid       ( new std::vector<int>() );
  std::auto_ptr<std::vector<int> >            side           ( new std::vector<int>() );
  std::auto_ptr<std::vector<int> >            module         ( new std::vector<int>() );
  std::auto_ptr<std::vector<int> >            layerwheel     ( new std::vector<int>() );
  std::auto_ptr<std::vector<int> >            stringringrod  ( new std::vector<int>() );
  std::auto_ptr<std::vector<int> >            petal          ( new std::vector<int>() );
  std::auto_ptr<std::vector<int> >            stereo         ( new std::vector<int>());

  edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusters;
  iEvent.getByLabel(theClustersLabel, clusters);
  
  edm::Handle<edm::DetSetVector<SiStripProcessedRawDigi> > rawProcessedDigis;
  iEvent.getByLabel("siStripProcessedRawDigis", "", rawProcessedDigis);
  
  BOOST_FOREACH( const edmNew::DetSet<SiStripCluster> ds, *clusters) {
    const moduleVars moduleV(ds.detId());
    BOOST_FOREACH( const SiStripCluster cluster, ds) {
      const SiStripClusterInfo info(cluster, iSetup);
      const NearDigis digis = rawProcessedDigis.isValid() ? NearDigis(info, *rawProcessedDigis) : NearDigis(info);

      (number->at(0))++;
      (number->at(moduleV.subdetid))++;
      width->push_back(        cluster.amplitudes().size()                              );
      barystrip->push_back(    cluster.barycenter()                                     );
      variance->push_back(     info.variance()                                         );
      middlestrip->push_back(  info.firstStrip() + info.width()/2.0                    );
      charge->push_back(       info.charge()                                           );
      noise->push_back(        info.noiseRescaledByGain()                              );
      ston->push_back(         info.signalOverNoise()                                  );
      seedstrip->push_back(    info.maxStrip()                                         );
      seedindex->push_back(    info.maxIndex()                                         );
      seedcharge->push_back(   info.maxCharge()                                        );
      seednoise->push_back(    info.stripNoisesRescaledByGain().at(info.maxIndex())   );
      seedgain->push_back(     info.stripGains().at(info.maxIndex())                  );
      qualityisbad->push_back( info.IsAnythingBad()                                    );

      rawchargeC->push_back(   digis.max            );
      rawchargeL->push_back(   digis.left           );
      rawchargeR->push_back(   digis.right          );
      rawchargeLL->push_back(  digis.Lleft          );
      rawchargeRR->push_back(  digis.Rright         );
      etaX->push_back(         digis.etaX()         );
      eta->push_back(          digis.eta()          );
      etaasymm->push_back(     digis.etaasymm()     );
      outsideasymm->push_back( digis.outsideasymm() );
      neweta->push_back(       (digis.last-digis.first)/info.charge() );
      newetaerr->push_back(    (sqrt(digis.last+digis.first))/pow(info.charge(),1.5) );

      detid->push_back(            ds.detId()            );                
      subdetid->push_back(         moduleV.subdetid      );          
      side->push_back(             moduleV.side          );                  
      module->push_back(           moduleV.module        );              
      layerwheel->push_back(       moduleV.layerwheel    );      
      stringringrod->push_back(    moduleV.stringringrod );
      petal->push_back(            moduleV.petal         );                
      stereo->push_back(           moduleV.stereo        );              
    }
  }

  iEvent.put( number,       Prefix + "number"       );
  iEvent.put( width,        Prefix + "width"        );
  iEvent.put( variance,     Prefix + "variance"     );
  iEvent.put( barystrip,    Prefix + "barystrip"    );
  iEvent.put( middlestrip,  Prefix + "middlestrip"  );
  iEvent.put( charge,       Prefix + "charge"       );
  iEvent.put( noise,        Prefix + "noise"        );
  iEvent.put( ston,         Prefix + "ston"         );
  iEvent.put( seedstrip,    Prefix + "seedstrip"    );
  iEvent.put( seedindex,    Prefix + "seedindex"    );
  iEvent.put( seedcharge,   Prefix + "seedcharge"   );
  iEvent.put( seednoise,    Prefix + "seednoise"    );
  iEvent.put( seedgain,     Prefix + "seedgain"     );
  iEvent.put( qualityisbad, Prefix + "qualityisbad" );

  iEvent.put( rawchargeC,   Prefix + "rawchargeC"   );
  iEvent.put( rawchargeL,   Prefix + "rawchargeL"   );
  iEvent.put( rawchargeR,   Prefix + "rawchargeR"   );
  iEvent.put( rawchargeLL,  Prefix + "rawchargeLL"  );
  iEvent.put( rawchargeRR,  Prefix + "rawchargeRR"  );
  iEvent.put( etaX,         Prefix + "etaX"         );
  iEvent.put( eta,          Prefix + "eta"          );
  iEvent.put( foldedeta,    Prefix + "foldedeta"    );
  iEvent.put( etaasymm,     Prefix + "etaasymm"     );
  iEvent.put( outsideasymm, Prefix + "outsideasymm" );
  iEvent.put( neweta,       Prefix + "neweta"       );
  iEvent.put( newetaerr,    Prefix + "newetaerr"    );

  iEvent.put( detid,         Prefix + "detid"         );
  iEvent.put( subdetid,      Prefix + "subdetid"      );
  iEvent.put( module,        Prefix + "module"        );
  iEvent.put( side,          Prefix + "side"          );
  iEvent.put( layerwheel,    Prefix + "layerwheel"    );
  iEvent.put( stringringrod, Prefix + "stringringrod" );
  iEvent.put( petal,         Prefix + "petal"         );
  iEvent.put( stereo,        Prefix + "stereo"        );

}

ShallowClustersProducer::NearDigis::
NearDigis(const SiStripClusterInfo& info) {
  max =  info.maxCharge();
  left =           info.maxIndex()    > uint16_t(0)                ? info.stripCharges().at(info.maxIndex()-1)      : 0 ;
  Lleft =          info.maxIndex()    > uint16_t(1)                ? info.stripCharges().at(info.maxIndex()-2)      : 0 ;
  right=  unsigned(info.maxIndex()+1) < info.stripCharges().size() ? info.stripCharges().at(info.maxIndex()+1)      : 0 ;
  Rright= unsigned(info.maxIndex()+2) < info.stripCharges().size() ? info.stripCharges().at(info.maxIndex()+2)      : 0 ;
  first = info.stripCharges().at(0);
  last =  info.stripCharges().at(info.width()-1);
}

ShallowClustersProducer::NearDigis::
NearDigis(const SiStripClusterInfo& info, const edm::DetSetVector<SiStripProcessedRawDigi>& rawProcessedDigis) {
  edm::DetSetVector<SiStripProcessedRawDigi>::const_iterator digiframe = rawProcessedDigis.find(info.detId());
  if( digiframe != rawProcessedDigis.end()) {
    max =                                                            digiframe->data.at(info.maxStrip()).adc()       ;
    left =            info.maxStrip()    > uint16_t(0)             ? digiframe->data.at(info.maxStrip()-1).adc() : 0 ;
    Lleft =           info.maxStrip()    > uint16_t(1)             ? digiframe->data.at(info.maxStrip()-2).adc() : 0 ;
    right =  unsigned(info.maxStrip()+1) < digiframe->data.size()  ? digiframe->data.at(info.maxStrip()+1).adc() : 0 ;
    Rright = unsigned(info.maxStrip()+2) < digiframe->data.size()  ? digiframe->data.at(info.maxStrip()+2).adc() : 0 ;
    first = digiframe->data.at(info.firstStrip()).adc();
    last = digiframe->data.at(info.firstStrip()+info.width() - 1).adc();
  } else {
    *this = NearDigis(info);
  }
}

ShallowClustersProducer::moduleVars::
moduleVars(uint32_t detid) {
  SiStripDetId subdet(detid);
  subdetid = subdet.subDetector();
  if( SiStripDetId::TIB == subdetid ) {
    TIBDetId tib(detid);
    module        = tib.module(); 
    side          = (tib.isZMinusSide())?-1:1;  
    layerwheel    = tib.layer(); 
    stringringrod = tib.stringNumber(); 
    stereo        = tib.isStereo() ? 1 : 0;
  } else
  if( SiStripDetId::TID == subdetid ) {
    TIDDetId tid(detid);
    module        = tid.moduleNumber(); 
    side          = (tid.isZMinusSide())?-1:1;  
    layerwheel    = tid.wheel(); 
    stringringrod = tid.ringNumber(); 
    stereo        = tid.isStereo() ? 1 : 0;
  } else
  if( SiStripDetId::TOB == subdetid ) {
    TOBDetId tob(detid);
    module        = tob.module(); 
    side          = (tob.isZMinusSide())?-1:1;  
    layerwheel    = tob.layer(); 
    stringringrod = tob.rodNumber(); 
    stereo        = tob.isStereo() ? 1 : 0;
  } else
  if( SiStripDetId::TEC == subdetid ) {
    TECDetId tec(detid);
    module        = tec.module(); 
    side          = (tec.isZMinusSide())?-1:1;  
    layerwheel    = tec.wheel(); 
    stringringrod = tec.ringNumber(); 
    petal         = tec.petalNumber(); 
    stereo        = tec.isStereo() ? 1 : 0;
  } else {
    module = 0;
    side = 0;
    layerwheel=-1;
    stringringrod = -1;
    petal=-1;
  }
}
