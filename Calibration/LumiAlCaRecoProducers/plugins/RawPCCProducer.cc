/**_________________________________________________________________
class:   RawPCCProducer.cc

description: Creates a LumiInfo object that will contain the luminosity per bunch crossing,
along with the total luminosity and the statistical error.

authors:Sam Higginbotham (shigginb@cern.ch) and Chris Palmer (capalmer@cern.ch) 

________________________________________________________________**/
#include <string>
#include <iostream> 
#include <fstream> 
#include <vector>
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
            ~RawPCCProducer() override;

        private:
            void beginLuminosityBlock     (edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) final;
            void endLuminosityBlock       (edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup) final;
            void endLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup) final;
            void produce                  (edm::Event& iEvent, const edm::EventSetup& iSetup) final;


            edm::EDGetTokenT<reco::PixelClusterCounts>  pccToken_;
            const edm::EDPutTokenT<LumiInfo> putToken_;
            const std::string   takeAverageValue_;      //Output average values 

            const std::vector<int>   modVeto_;          //The list of modules to skip in the lumi calc. 

            const std::string csvOutLabel_;
            const bool saveCSVFile_;

            const bool applyCorr_;

            std::ofstream csvfile_;
    };

//--------------------------------------------------------------------------------------------------
RawPCCProducer::RawPCCProducer(const edm::ParameterSet& iConfig):
  putToken_{ produces<LumiInfo, edm::Transition::EndLuminosityBlock>(iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getUntrackedParameter<std::string>("outputProductName","alcaLumi"))},
  takeAverageValue_ {iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getUntrackedParameter<std::string>("OutputValue",std::string("Totals"))},
  modVeto_{ iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getParameter<std::vector<int>>("modVeto")},
  csvOutLabel_{ iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getUntrackedParameter<std::string>("label",std::string("rawPCC.csv")) },
  saveCSVFile_{ iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getUntrackedParameter<bool>("saveCSVFile",false)},
  applyCorr_{ iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getUntrackedParameter<bool>("ApplyCorrections",false) }
{
    auto pccSource = iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getParameter<std::string>("inputPccLabel");
    auto prodInst = iConfig.getParameter<edm::ParameterSet>("RawPCCProducerParameters").getParameter<std::string>("ProdInst");

    pccToken_=consumes<reco::PixelClusterCounts, edm::InLumi>(edm::InputTag(pccSource, prodInst));
}

//--------------------------------------------------------------------------------------------------
RawPCCProducer::~RawPCCProducer(){
}

//--------------------------------------------------------------------------------------------------
void RawPCCProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){


}

//--------------------------------------------------------------------------------------------------
void RawPCCProducer::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup){


    if(saveCSVFile_){
        csvfile_.open(csvOutLabel_, std::ios_base::app);
        csvfile_<<std::to_string(lumiSeg.run())<<",";
        csvfile_<<std::to_string(lumiSeg.luminosityBlock())<<",";
    }
}

//--------------------------------------------------------------------------------------------------
void RawPCCProducer::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, const edm::EventSetup& iSetup){
}

//--------------------------------------------------------------------------------------------------
void RawPCCProducer::endLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup){
    float totalLumi=0.0; //The total raw luminosity from the pixel clusters - not scaled
    float statErrOnLumi=0.0; //the statistical error on the lumi - large num ie sqrt(N)

    //new vector containing clusters per bxid
    std::vector<int> clustersPerBXOutput(LumiConstants::numBX,0);
    //new vector containing clusters per bxid with afterglow corrections 
    std::vector<float> corrClustersPerBXOutput(LumiConstants::numBX,0);

    //The indicies of all the good modules - not vetoed
    std::vector<int> goodMods;                 

    edm::Handle<reco::PixelClusterCounts> pccHandle; 
    lumiSeg.getByToken(pccToken_,pccHandle);

    const reco::PixelClusterCounts& inputPcc = *(pccHandle.product()); 

    //vector with Module IDs 1-1 map to bunch x-ing in clusers
    auto modID = inputPcc.readModID();
    //vector with total events at each bxid.
    auto events= inputPcc.readEvents();
    auto clustersPerBXInput = inputPcc.readCounts();

    //making list of modules to sum over
    for (unsigned int i=0;i<modID.size();i++){
        if (std::find(modVeto_.begin(),modVeto_.end(), modID.at(i)) == modVeto_.end()){
            goodMods.push_back(i);
        }
    }

    //summing over good modules only 
    for (int bx=0;bx<int(LumiConstants::numBX);bx++){
        for (unsigned int i=0;i<goodMods.size();i++){
            clustersPerBXOutput.at(bx)+=clustersPerBXInput.at(goodMods.at(i)*int(LumiConstants::numBX)+bx);

        }
    }

    std::vector<float> correctionScaleFactors;
    if(applyCorr_){
        edm::ESHandle< LumiCorrections > corrHandle;
        iSetup.get<LumiCorrectionsRcd>().get(corrHandle);
        const LumiCorrections *pccCorrections = corrHandle.product();
        correctionScaleFactors = pccCorrections->getCorrectionsBX();
    } else {
        correctionScaleFactors.resize(LumiConstants::numBX,1.0);
    }

    for (unsigned int i=0;i<clustersPerBXOutput.size();i++){
        if (events.at(i)!=0){
            corrClustersPerBXOutput[i]=clustersPerBXOutput[i]*correctionScaleFactors[i];
        }else{
            corrClustersPerBXOutput[i]=0.0;
        }
        totalLumi+=corrClustersPerBXOutput[i];
        statErrOnLumi+=float(events[i]);
    }

    std::vector<float> errorPerBX;             //Stat error (or number of events)
    errorPerBX.assign(events.begin(),events.end()); 

    if(takeAverageValue_=="Average"){
        unsigned int NActiveBX=0;
        for (int bx=0;bx<int(LumiConstants::numBX);bx++){
            if(events[bx]>0){
                NActiveBX++;  
                // Counting where events are will only work 
                // for ZeroBias or AlwaysTrue triggers.
                // Random triggers will get all BXs.
                corrClustersPerBXOutput[bx]/=float(events[bx]);
                errorPerBX[bx]=1/TMath::Sqrt(float(events[bx]));
            }
        }
        if (statErrOnLumi!=0) {
            totalLumi=totalLumi/statErrOnLumi*float(NActiveBX);
            statErrOnLumi=1/TMath::Sqrt(statErrOnLumi)*totalLumi;
        }
    }

    LumiInfo outputLumiInfo;


    outputLumiInfo.setTotalInstLumi(totalLumi);
    outputLumiInfo.setTotalInstStatError(statErrOnLumi);

    outputLumiInfo.setErrorLumiAllBX(errorPerBX);
    outputLumiInfo.setInstLumiAllBX(corrClustersPerBXOutput);

    if(saveCSVFile_){
        csvfile_<<std::to_string(totalLumi);
        
        if(totalLumi>0){
            for(unsigned int bx=0;bx<LumiConstants::numBX;bx++){
                csvfile_<<","<<std::to_string(corrClustersPerBXOutput[bx]);
            }
            csvfile_<<std::endl;   
        } else if (totalLumi<0) {
            edm::LogInfo("WARNING")<<"WHY IS LUMI NEGATIVE?!?!?!? "<<totalLumi;
        }

        csvfile_.close();
    }
    lumiSeg.emplace(putToken_, std::move(outputLumiInfo) ); 

}

DEFINE_FWK_MODULE(RawPCCProducer);
