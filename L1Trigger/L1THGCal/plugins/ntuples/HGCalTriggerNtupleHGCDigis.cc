#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerNtupleBase.h"



class HGCalTriggerNtupleHGCDigis : public HGCalTriggerNtupleBase
{

    public:
        HGCalTriggerNtupleHGCDigis(const edm::ParameterSet& conf);
        ~HGCalTriggerNtupleHGCDigis(){};
        virtual void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) override final;
        virtual void fill(const edm::Event& e, const edm::EventSetup& es) override final;

    private:
        virtual void clear() override final;

        edm::EDGetToken ee_token_, fh_token_;

        int hgcdigi_n_ ;
        std::vector<int> hgcdigi_id_;
        std::vector<int> hgcdigi_subdet_;
        std::vector<int> hgcdigi_side_;
        std::vector<int> hgcdigi_layer_;
        std::vector<int> hgcdigi_wafer_;
        std::vector<int> hgcdigi_wafertype_ ;
        std::vector<int> hgcdigi_cell_;
        std::vector<uint32_t> hgcdigi_data_;

};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory,
        HGCalTriggerNtupleHGCDigis,
        "HGCalTriggerNtupleHGCDigis" );


HGCalTriggerNtupleHGCDigis::
HGCalTriggerNtupleHGCDigis(const edm::ParameterSet& conf):HGCalTriggerNtupleBase(conf)
{
}

void
HGCalTriggerNtupleHGCDigis::
initialize(TTree& tree, const edm::ParameterSet& conf, edm::ConsumesCollector&& collector)
{

    ee_token_ = collector.consumes<HGCEEDigiCollection>(conf.getParameter<edm::InputTag>("HGCDigisEE")); 
    fh_token_ = collector.consumes<HGCHEDigiCollection>(conf.getParameter<edm::InputTag>("HGCDigisFH"));

    tree.Branch("hgcdigi_n", &hgcdigi_n_, "hgcdigi_n/I");
    tree.Branch("hgcdigi_id", &hgcdigi_id_);
    tree.Branch("hgcdigi_subdet", &hgcdigi_subdet_);
    tree.Branch("hgcdigi_zside", &hgcdigi_side_);
    tree.Branch("hgcdigi_layer", &hgcdigi_layer_);
    tree.Branch("hgcdigi_wafer", &hgcdigi_wafer_);
    tree.Branch("hgcdigi_wafertype", &hgcdigi_wafertype_);
    tree.Branch("hgcdigi_cell", &hgcdigi_cell_);    
    tree.Branch("hgcdigi_data", &hgcdigi_data_);

}

void
HGCalTriggerNtupleHGCDigis::
fill(const edm::Event& e, const edm::EventSetup& es)
{
    edm::Handle<HGCEEDigiCollection> ee_digis_h;
    edm::Handle<HGCHEDigiCollection> fh_digis_h;

    e.getByToken(ee_token_, ee_digis_h);
    e.getByToken(fh_token_, fh_digis_h);

    const HGCEEDigiCollection& ee_digis = *ee_digis_h;
    const HGCHEDigiCollection& fh_digis = *fh_digis_h;

    clear();
    hgcdigi_n_ = ee_digis.size() + fh_digis.size();
    hgcdigi_id_.reserve(hgcdigi_n_);
    hgcdigi_subdet_.reserve(hgcdigi_n_);
    hgcdigi_side_.reserve(hgcdigi_n_);
    hgcdigi_layer_.reserve(hgcdigi_n_);
    hgcdigi_wafer_.reserve(hgcdigi_n_);
    hgcdigi_wafertype_.reserve(hgcdigi_n_);
    hgcdigi_cell_.reserve(hgcdigi_n_);
    hgcdigi_data_.reserve(hgcdigi_n_);
    for(const auto& digi : ee_digis)
    {
        const HGCalDetId id(digi.id());

        hgcdigi_id_.emplace_back(id.rawId());
        hgcdigi_subdet_.emplace_back(ForwardSubdetector::HGCEE);
        hgcdigi_side_.emplace_back(id.zside());
        hgcdigi_layer_.emplace_back(id.layer());
        hgcdigi_wafer_.emplace_back(id.wafer());
        hgcdigi_wafertype_.emplace_back(id.waferType());
        hgcdigi_cell_.emplace_back(id.cell());
        hgcdigi_data_.emplace_back(digi[2].raw()); // in-time
    }
    for(const auto& digi : fh_digis)
    {
        const HGCalDetId id(digi.id());

        hgcdigi_id_.emplace_back(id.rawId());
        hgcdigi_subdet_.emplace_back(ForwardSubdetector::HGCHEF);
        hgcdigi_side_.emplace_back(id.zside());
        hgcdigi_layer_.emplace_back(id.layer());
        hgcdigi_wafer_.emplace_back(id.wafer());
        hgcdigi_wafertype_.emplace_back(id.waferType());
        hgcdigi_cell_.emplace_back(id.cell());
        hgcdigi_data_.emplace_back(digi[2].raw()); // in-time
    }
}


void
HGCalTriggerNtupleHGCDigis::
clear()
{
    hgcdigi_n_ = 0;
    hgcdigi_id_.clear();
    hgcdigi_subdet_.clear();
    hgcdigi_side_.clear();
    hgcdigi_layer_.clear();
    hgcdigi_wafer_.clear();
    hgcdigi_wafertype_.clear();
    hgcdigi_cell_.clear();
    hgcdigi_data_.clear();
}




