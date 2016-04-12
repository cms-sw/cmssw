#include <iostream>
#include <string>
#include <vector>


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/L1THGCal/interface/HGCFETriggerDigi.h"
#include "DataFormats/L1THGCal/interface/HGCFETriggerDigiFwd.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"

#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerFECodecBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalBestChoiceCodecImpl.h"

#include <stdlib.h> 


class HGCalTriggerBestChoiceTester : public edm::EDAnalyzer 
{
    public:
        explicit HGCalTriggerBestChoiceTester(const edm::ParameterSet& );
        ~HGCalTriggerBestChoiceTester();

        virtual void beginRun(const edm::Run&, const edm::EventSetup&);
        virtual void analyze(const edm::Event&, const edm::EventSetup&);


    private:
        void fillModule(const std::vector<HGCEEDataFrame>&, const HGCalBestChoiceDataPayload&);
        // inputs
        edm::EDGetToken inputee_, inputfh_, inputbh_;
        //
        std::unique_ptr<HGCalTriggerGeometryBase> triggerGeometry_; 
        std::unique_ptr<HGCalBestChoiceCodecImpl> codec_;
        edm::Service<TFileService> fs_;
        // histos
        TH1F* hgcCellsPerModule_;
        TH1F* hgcCellData_;
        TH1F* hgcCellModuleSum_;
        TH1F* triggerCellsPerModule_;
        TH1F* triggerCellData_;
        TH1F* triggerCellModuleSum_;

};


/*****************************************************************/
HGCalTriggerBestChoiceTester::HGCalTriggerBestChoiceTester(const edm::ParameterSet& conf):
  inputee_(consumes<HGCEEDigiCollection>(conf.getParameter<edm::InputTag>("eeDigis"))),
  inputfh_(consumes<HGCHEDigiCollection>(conf.getParameter<edm::InputTag>("fhDigis"))), 
  inputbh_(consumes<HGCHEDigiCollection>(conf.getParameter<edm::InputTag>("bhDigis")))
/*****************************************************************/
{
    //setup geometry 
    const edm::ParameterSet& geometryConfig = conf.getParameterSet("TriggerGeometry");
    const std::string& trigGeomName = geometryConfig.getParameter<std::string>("TriggerGeometryName");
    HGCalTriggerGeometryBase* geometry = HGCalTriggerGeometryFactory::get()->create(trigGeomName,geometryConfig);
    triggerGeometry_.reset(geometry);

    //setup FE codec
    const edm::ParameterSet& feCodecConfig = conf.getParameterSet("FECodec");
    codec_.reset( new HGCalBestChoiceCodecImpl(feCodecConfig) );

    // initialize output trees
    hgcCellsPerModule_     = fs_->make<TH1F>("hgcCellsPerModule","Number of cells per module", 64, 0., 64.);
    hgcCellData_           = fs_->make<TH1F>("hgcCellData","Cell values", 500, 0., 500.);
    hgcCellModuleSum_      = fs_->make<TH1F>("hgcCellModuleSum","Cell sum in modules", 1000, 0., 1000.);
    triggerCellsPerModule_ = fs_->make<TH1F>("TriggerCellsPerModule","Number of trigger cells per module", 64, 0., 64.);
    triggerCellData_       = fs_->make<TH1F>("TriggerCellData","Trigger cell values", 500, 0., 500.);
    triggerCellModuleSum_  = fs_->make<TH1F>("TriggerCellModuleSum","Trigger cell sum in modules", 1000, 0., 1000.);
}



/*****************************************************************/
HGCalTriggerBestChoiceTester::~HGCalTriggerBestChoiceTester() 
/*****************************************************************/
{
}

/*****************************************************************/
void HGCalTriggerBestChoiceTester::beginRun(const edm::Run& /*run*/, 
                                          const edm::EventSetup& es)
/*****************************************************************/
{
    triggerGeometry_->reset();
    HGCalTriggerGeometryBase::es_info info;
    const std::string& ee_sd_name = triggerGeometry_->eeSDName();
    const std::string& fh_sd_name = triggerGeometry_->fhSDName();
    const std::string& bh_sd_name = triggerGeometry_->bhSDName();
    es.get<IdealGeometryRecord>().get(ee_sd_name,info.geom_ee);
    es.get<IdealGeometryRecord>().get(fh_sd_name,info.geom_fh);
    es.get<IdealGeometryRecord>().get(bh_sd_name,info.geom_bh);
    es.get<IdealGeometryRecord>().get(ee_sd_name,info.topo_ee);
    es.get<IdealGeometryRecord>().get(fh_sd_name,info.topo_fh);
    es.get<IdealGeometryRecord>().get(bh_sd_name,info.topo_bh);
    triggerGeometry_->initialize(info);
}



/*****************************************************************/
void HGCalTriggerBestChoiceTester::analyze(const edm::Event& e, 
			      const edm::EventSetup& es) 
/*****************************************************************/
{
    edm::Handle<HGCEEDigiCollection> ee_digis_h;
    e.getByToken(inputee_,ee_digis_h);

    const HGCEEDigiCollection& ee_digis = *ee_digis_h;

    HGCalBestChoiceDataPayload data;
    for( const auto& module : triggerGeometry_->modules() ) 
    {    
        // prepare input data
        std::vector<HGCEEDataFrame> dataframes;
        for(const auto& eedata : ee_digis)
        {
            if(module.second->containsCell(eedata.id()))
            {
                dataframes.push_back(eedata);
            }
        }
        // Best choice encoding
        data.reset();
        codec_->triggerCellSums(*(module.second), dataframes, data);
        codec_->bestChoiceSelect(data);
        std::vector<bool> dataword = codec_->encode(data);
        HGCalBestChoiceDataPayload datadecoded = codec_->decode(dataword);
        fillModule(dataframes, datadecoded);
    }

}


/*****************************************************************/
void HGCalTriggerBestChoiceTester::fillModule( const std::vector<HGCEEDataFrame>& dataframes, const HGCalBestChoiceDataPayload& fe_payload)
/*****************************************************************/
{
    // HGC cells part
    size_t nHGCDigi = 0;
    unsigned hgcCellModuleSum = 0;
    for(const auto& frame : dataframes)
    {
        uint32_t value = frame[2].data();
        if(value>0)
        {
            nHGCDigi++;
            hgcCellModuleSum += value;
            hgcCellData_->Fill(value);
        }
    }
    hgcCellsPerModule_->Fill(nHGCDigi);
    hgcCellModuleSum_->Fill(hgcCellModuleSum);

    // trigger cells part
    size_t nFEDigi = 0;
    unsigned triggerCellModuleSum = 0;
    for(const auto& tc : fe_payload.payload)
    {
        uint32_t tcShifted = (tc<<2);
        if(tcShifted>0)
        {
            nFEDigi++;
            triggerCellModuleSum += tcShifted;
            triggerCellData_->Fill(tcShifted);
        }
    }
    triggerCellsPerModule_->Fill(nFEDigi);
    triggerCellModuleSum_->Fill(triggerCellModuleSum);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTriggerBestChoiceTester);
