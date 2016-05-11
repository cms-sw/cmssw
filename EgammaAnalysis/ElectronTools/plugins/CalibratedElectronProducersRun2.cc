#ifndef CalibratedElectronProducerRun2_h
#define CalibratedElectronProducerRun2_h

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "EgammaAnalysis/ElectronTools/interface/EpCombinationTool.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEnergyCalibratorRun2.h"

#include <vector>

template<typename T>
class CalibratedElectronProducerRun2T: public edm::stream::EDProducer<>
{
    public:
        explicit CalibratedElectronProducerRun2T( const edm::ParameterSet & ) ;
        virtual ~CalibratedElectronProducerRun2T();
        virtual void produce( edm::Event &, const edm::EventSetup & ) override ;

    private:
        edm::EDGetTokenT<edm::View<T> > theElectronToken;
        std::string theGBRForestName;
        edm::ESHandle<GBRForest> theGBRForestHandle;

        EpCombinationTool theEpCombinationTool;
        ElectronEnergyCalibratorRun2 theEnCorrectorRun2;
};

template<typename T>
CalibratedElectronProducerRun2T<T>::CalibratedElectronProducerRun2T( const edm::ParameterSet & conf ) :
  theElectronToken(consumes<edm::View<T> >(conf.getParameter<edm::InputTag>("electrons"))),
  theGBRForestName(conf.getParameter<std::string>("gbrForestName")),
  theEpCombinationTool(),
  theEnCorrectorRun2(theEpCombinationTool, conf.getParameter<bool>("isMC"), conf.getParameter<bool>("isSynchronization"), conf.getParameter<std::string>("correctionFile"))
{
  produces<std::vector<T> >();
}

template<typename T>
CalibratedElectronProducerRun2T<T>::~CalibratedElectronProducerRun2T()
{
}

template<typename T>
void
CalibratedElectronProducerRun2T<T>::produce( edm::Event & iEvent, const edm::EventSetup & iSetup ) 
{
    iSetup.get<GBRWrapperRcd>().get(theGBRForestName, theGBRForestHandle);
    theEpCombinationTool.init(theGBRForestHandle.product() );

    edm::Handle<edm::View<T> > in;
    iEvent.getByToken(theElectronToken, in);

    std::auto_ptr<std::vector<T> > out(new std::vector<T>());
    out->reserve(in->size());   

    for (const T &ele : *in) {
        out->push_back(ele);
        theEnCorrectorRun2.calibrate(out->back(), iEvent.id().run(), iEvent.streamID());
    }
    
    iEvent.put(out);
}

typedef CalibratedElectronProducerRun2T<reco::GsfElectron> CalibratedElectronProducerRun2;
typedef CalibratedElectronProducerRun2T<pat::Electron> CalibratedPatElectronProducerRun2;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CalibratedElectronProducerRun2);
DEFINE_FWK_MODULE(CalibratedPatElectronProducerRun2);

#endif
