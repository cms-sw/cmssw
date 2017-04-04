/**_________________________________________________________________
class:   AlcaPCCProducer.cc



authors:Sam Higginbotham (shigginb@cern.ch) and Chris Palmer (capalmer@cern.ch) 

________________________________________________________________**/


// C++ standard
#include <string>
// CMS
#include "DataFormats/Luminosity/interface/PCC.h"
#include "Calibration/PCCAlCaRecoProducers/interface/AlcaPCCProducer.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "TMath.h"

//--------------------------------------------------------------------------------------------------
AlcaPCCProducer::AlcaPCCProducer(const edm::ParameterSet& iConfig)
{
    resetNLumi_ = iConfig.getParameter<edm::ParameterSet>("AlcaPCCProducerParameters").getUntrackedParameter<int>("resetEveryNLumi",-1);
    fPixelClusterLabel = iConfig.getParameter<edm::ParameterSet>("AlcaPCCProducerParameters").getParameter<edm::InputTag>("pixelClusterLabel");
    trigstring_ = iConfig.getParameter<edm::ParameterSet>("AlcaPCCProducerParameters").getUntrackedParameter<std::string>("trigstring","alcaPCC");

    //std::cout<<"A Print Statement"<<std::endl;
    ftotalevents = 0;
    countLumi_ = 0;
    beginLumiOfPCC_ = endLumiOfPCC_ = -1;

    produces<reco::PCC, edm::InLumi>(trigstring_);
    pixelToken=consumes<edmNew::DetSetVector<SiPixelCluster> >(fPixelClusterLabel);
}

//--------------------------------------------------------------------------------------------------
AlcaPCCProducer::~AlcaPCCProducer(){
}

//--------------------------------------------------------------------------------------------------
void AlcaPCCProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
    ftotalevents++;

    unsigned int bx=iEvent.bunchCrossing();
    //std::cout<<"The Bunch Crossing"<<bx<<std::endl;
    thePCCob->eventCounter(bx);

    //Looping over the clusters and adding the counts up  
    edm::Handle< edmNew::DetSetVector<SiPixelCluster> > hClusterColl;
    iEvent.getByToken(pixelToken,hClusterColl);

    const edmNew::DetSetVector<SiPixelCluster>& clustColl = *(hClusterColl.product()); 
    // ----------------------------------------------------------------------
    // -- Clusters without tracks
    for (auto const & mod: clustColl) {
        if(mod.empty()) { continue; }
        DetId detId = mod.id();

        // -- clusters on this det
        edmNew::DetSet<SiPixelCluster>::const_iterator  di;
        int nClusterCount=0;
        for (di = mod.begin(); di != mod.end(); ++di) {
            nClusterCount++;
        }
        int nCluster = mod.size();
        if(nCluster!=nClusterCount) {
            std::cout<<"counting yields "<<nClusterCount<<" but the size is "<<nCluster<<"; they should match."<<std::endl;
        }
        thePCCob->Increment(detId(), bx, nCluster);
    }
}

//--------------------------------------------------------------------------------------------------
void AlcaPCCProducer::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup){
    //New PCC object at the beginning of each lumi section
    thePCCob = std::make_unique<reco::PCC>();
    
    if ( countLumi_ == 0 || (resetNLumi_ > 0 && countLumi_%resetNLumi_ == 0) ) {
        beginLumiOfPCC_ = lumiSeg.luminosityBlock();
    }

    countLumi_++;

}

//--------------------------------------------------------------------------------------------------
void AlcaPCCProducer::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup){
    //Saving the PCC object 
    //FIXME! The line below doesn't work but I need to save here. 
    //lumiSeg.put(std::move(thePCCob), std::string(trigstring_)); 
}

//--------------------------------------------------------------------------------------------------
void AlcaPCCProducer::endLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup){

    endLumiOfPCC_ = lumiSeg.luminosityBlock();

    if (resetNLumi_ == -1) return;

    if (countLumi_%resetNLumi_!=0) return;

    //Saving the PCC object 
    std::cout<<"Saving Object "<<std::endl;
    lumiSeg.put(std::move(thePCCob), std::string(trigstring_)); 

}

DEFINE_FWK_MODULE(AlcaPCCProducer);
