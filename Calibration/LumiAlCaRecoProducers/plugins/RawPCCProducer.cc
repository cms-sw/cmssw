/**_________________________________________________________________
class:   RawPCCProducer.cc

description: Creates a LumiInfo object that will contain the luminosity per bunch crossing,
along with the total luminosity and the statistical error.

authors:Sam Higginbotham (shigginb@cern.ch) and Chris Palmer (capalmer@cern.ch) 

________________________________________________________________**/


// C++ standard
#include <string>
#include <iostream> 
#include <fstream> 
#include <vector>
// CMS
#include "DataFormats/Luminosity/interface/PixelClusterCounts.h"
#include "DataFormats/Luminosity/interface/LumiInfo.h"
#include "DataFormats/Luminosity/interface/LumiConstants.h"
#include "CondFormats/Luminosity/interface/LumiCorrections.h"
#include "CondFormats/DataRecord/interface/LumiCorrectionsRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "TMath.h"

class RawPCCProducer : public edm::one::EDProducer<edm::EndLuminosityBlockProducer,
    edm::one::WatchLuminosityBlocks> {
        public:
            explicit RawPCCProducer(const edm::ParameterSet&);
            ~RawPCCProducer();

        private:
            virtual void beginLuminosityBlock     (edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) override final;
            virtual void endLuminosityBlock       (edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) override final;
            virtual void endLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup) override final;
            virtual void produce                  (edm::Event& iEvent, const edm::EventSetup& iSetup) override final;

            edm::EDGetTokenT<reco::PixelClusterCounts>  pccToken;
            std::string   pccSource_;//input file EDproducer module label 
            std::string   prodInst_;//input file product instance 
            std::string   takeAverageValue_;//Output average values 

            std::vector<int>   modVeto_;//The list of modules to skip in the lumi calc. 
            std::string trigstring_; //specifies the trigger Rand or ZeroBias 
            std::vector<int> clusters_;//Will fill this with content from PCC
            std::vector<int> modID_;//vector with Module IDs 1-1 map to bunch x-ing in clusers_
            std::vector<int> events_;//vector with total events at each bxid.
            std::vector<int> clusterPerBX_;//new vector containing clusters per bxid 
            std::vector<float> rawlumiBX_;//new vector containing clusters per bxid 
            std::vector<float> errOnLumiByBX_;//standard error per bx
            std::vector<int> goodMods_;//The indicies of all the good modules - not vetoed
            float totalLumi_;//The total raw luminosity from the pixel clusters - not scaled
            float statErrOnLumi_;//the statistical error on the lumi - large num ie sqrt(N)

            std::string csvOutLabel_;

            bool applyCorr_;
            std::vector<float> correctionScaleFactors_;

            std::unique_ptr<LumiInfo> theLumiOb;

            std::ofstream csvfile;
    };

//--------------------------------------------------------------------------------------------------
RawPCCProducer::RawPCCProducer(const edm::ParameterSet& iConfig)
{
    pccSource_ = iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getParameter<std::string>("PCCobLabel");
    prodInst_ = iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getParameter<std::string>("ProdInst");
    takeAverageValue_ = iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getUntrackedParameter<std::string>("OutputValue",std::string("Totals")); 
    trigstring_ = iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getUntrackedParameter<std::string>("trigstring","alcaLumi");
    modVeto_ = iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getParameter<std::vector<int>>("modVeto");
    applyCorr_ = iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getUntrackedParameter<bool>("ApplyCorrections",false);
    csvOutLabel_ = iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getUntrackedParameter<std::string>("label",std::string("rawPCC.csv"));

    edm::InputTag PCCInputTag_(pccSource_, prodInst_);
    clusterPerBX_.resize(LumiConstants::numBX,0);//vector containing clusters per bxid 
    rawlumiBX_.resize(LumiConstants::numBX,0);//vector containing clusters per bxid 
    errOnLumiByBX_.resize(LumiConstants::numBX,0);
    pccToken=consumes<reco::PixelClusterCounts, edm::InLumi>(PCCInputTag_);
    produces<LumiInfo, edm::Transition::EndLuminosityBlock>(trigstring_);

    if(!applyCorr_){
        correctionScaleFactors_.resize(LumiConstants::numBX,1.0);
    }
}

//--------------------------------------------------------------------------------------------------
RawPCCProducer::~RawPCCProducer(){
}

//--------------------------------------------------------------------------------------------------
void RawPCCProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){


}

//--------------------------------------------------------------------------------------------------
void RawPCCProducer::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup){
    theLumiOb = std::make_unique<LumiInfo>(); 

    csvfile.open(csvOutLabel_, std::ios_base::app);
    csvfile<<std::to_string(lumiSeg.run())<<",";
    csvfile<<std::to_string(lumiSeg.luminosityBlock())<<",";
}

//--------------------------------------------------------------------------------------------------
void RawPCCProducer::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup){
    totalLumi_=0.0;
    statErrOnLumi_=0.0;

    std::fill(clusterPerBX_.begin(),  clusterPerBX_.end(),0);
    std::fill(rawlumiBX_.begin(),     rawlumiBX_.end(),   0);
    std::fill(errOnLumiByBX_.begin(), errOnLumiByBX_.end(),0);

    goodMods_.clear();

    edm::Handle<reco::PixelClusterCounts> pccHandle; 
    lumiSeg.getByToken(pccToken,pccHandle);

    const reco::PixelClusterCounts& PCCob = *(pccHandle.product()); 

    modID_ = PCCob.readModID();
    events_= PCCob.readEvents();
    clusters_ = PCCob.readCounts();


    for (unsigned int i=0;i<modID_.size();i++){
        if (std::find(modVeto_.begin(),modVeto_.end(), modID_.at(i)) == modVeto_.end()){
            goodMods_.push_back(i);
        }
    }

    //summing over modules
    for (int bx=0;bx<int(LumiConstants::numBX);bx++){
        for (unsigned int i=0;i<goodMods_.size();i++){
            clusterPerBX_.at(bx)+=clusters_.at(goodMods_.at(i)*int(LumiConstants::numBX)+bx);

        }
        if (clusterPerBX_.at(bx)!=0){
            errOnLumiByBX_.at(bx)=1/TMath::Sqrt(clusterPerBX_.at(bx));
        }
        else{
            errOnLumiByBX_.at(bx)=0.0;
        }
    }

    if(applyCorr_){
        edm::ESHandle< LumiCorrections > corrHandle;
        iSetup.get<LumiCorrectionsRcd>().get(corrHandle);
        const LumiCorrections *pccCorrections = corrHandle.product();
        correctionScaleFactors_ = pccCorrections->getCorrectionsBX();
    }

    for (unsigned int i=0;i<clusterPerBX_.size();i++){
        if (events_.at(i)!=0){
            rawlumiBX_[i]=clusterPerBX_[i]*correctionScaleFactors_[i];
        }else{
            rawlumiBX_[i]=0.0;
        }
        totalLumi_+=rawlumiBX_[i];
        statErrOnLumi_+=float(events_[i]);
    }

    std::vector<float> errorPerBX(events_.begin(), events_.end());

    if(takeAverageValue_=="Average"){
        unsigned int NActiveBX=0;
        for (int bx=0;bx<int(LumiConstants::numBX);bx++){
            if(events_[bx]>0){
                NActiveBX++;  
                // Counting where events are will only work 
                // for ZeroBias or AlwaysTrue triggers.
                // Random triggers will get all BXs.
                rawlumiBX_[bx]/=float(events_[bx]);
                errorPerBX[bx]=1/TMath::Sqrt(float(events_[bx]));
            }
        }
        if (statErrOnLumi_!=0) {
            totalLumi_=totalLumi_/statErrOnLumi_*float(NActiveBX);
            statErrOnLumi_=1/TMath::Sqrt(statErrOnLumi_)*totalLumi_;
        }
    }


    theLumiOb->setTotalInstLumi(totalLumi_);
    theLumiOb->setTotalInstStatError(statErrOnLumi_);

    theLumiOb->setErrorLumiAllBX(errorPerBX);
    theLumiOb->setInstLumiAllBX(rawlumiBX_);

    csvfile<<std::to_string(totalLumi_);

    if(totalLumi_>0){
        for(unsigned int bx=0;bx<LumiConstants::numBX;bx++){
            csvfile<<","<<std::to_string(rawlumiBX_[bx]);
        }
        csvfile<<std::endl;   
    } else if (totalLumi_<0) {
        edm::LogInfo("WARNING")<<"WHY IS LUMI NEGATIVE?!?!?!? "<<totalLumi_;
    }

    csvfile.close();
}

//--------------------------------------------------------------------------------------------------
void RawPCCProducer::endLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup){
    lumiSeg.put(std::move(theLumiOb), std::string(trigstring_)); 

}

DEFINE_FWK_MODULE(RawPCCProducer);
