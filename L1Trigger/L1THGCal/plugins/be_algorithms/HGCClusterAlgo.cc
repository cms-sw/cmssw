#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendAlgorithmBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodec.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellThresholdCodec.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTriggerCellCalibration.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalClusteringImpl.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalMulticlusteringImpl.h"

using namespace HGCalTriggerBackend;

template<typename FECODEC, typename DATA>
class HGCClusterAlgo : public Algorithm<FECODEC> 
{
public:
    using Algorithm<FECODEC>::name;
    
protected:
    using Algorithm<FECODEC>::codec_;
    
public:

    HGCClusterAlgo(const edm::ParameterSet& conf, edm::ConsumesCollector &cc) :
        Algorithm<FECODEC>(conf, cc),
        trgcell_product_( new l1t::HGCalTriggerCellBxCollection ),
        cluster_product_( new l1t::HGCalClusterBxCollection ),
        multicluster_product_( new l1t::HGCalMulticlusterBxCollection ),
        HGCalEESensitive_( conf.getParameter<std::string>("HGCalEESensitive_tag") ),
        HGCalHESiliconSensitive_( conf.getParameter<std::string>("HGCalHESiliconSensitive_tag") ),
        calibration_( conf.getParameterSet("calib_parameters") ),
        clustering_( conf.getParameterSet("C2d_parameters") ),
        multiclustering_( conf.getParameterSet("C3d_parameters" ) ) 
        {
        }
            
    typedef std::unique_ptr<HGCalTriggerGeometryBase> ReturnType;
    
    virtual void setProduces(edm::EDProducer& prod) const override final 
        {
            prod.produces<l1t::HGCalClusterBxCollection>(name());
            prod.produces<l1t::HGCalMulticlusterBxCollection>("cluster3D");
            
        }
    
    
    virtual void run(const l1t::HGCFETriggerDigiCollection& coll, const edm::EventSetup& es, const edm::Event&evt ) override final;
    virtual void putInEvent(edm::Event& evt) override final 
        {
            evt.put(std::move(trgcell_product_),name());
            evt.put(std::move(cluster_product_), name());
            evt.put(std::move(multicluster_product_), "cluster3D");

        }
    
    virtual void reset() override final 
        {
            trgcell_product_->clear();
            cluster_product_->clear();            
            multicluster_product_->clear();            
        }
    
private:
    
    std::unique_ptr<l1t::HGCalTriggerCellBxCollection> trgcell_product_;
    std::unique_ptr<l1t::HGCalClusterBxCollection> cluster_product_;
    std::unique_ptr<l1t::HGCalMulticlusterBxCollection> multicluster_product_;
    
    std::string HGCalEESensitive_;
    std::string HGCalHESiliconSensitive_;
    
    edm::ESHandle<HGCalTopology> hgceeTopoHandle_;
    edm::ESHandle<HGCalTopology> hgchefTopoHandle_;
    HGCalTriggerCellCalibration calibration_;    
    double seed_CUT_;
    double tc_CUT_;
    HGCalClusteringImpl clustering_;     
    HGCalMulticlusteringImpl multiclustering_;     
    double dR_forC3d_;
    
    double deltaPhi( double phi1, double phi2) {
        
        double dPhi(phi1-phi2);
        double pi(acos(-1.0));
        if     (dPhi<=-pi) dPhi+=2.0*pi;
        else if(dPhi> pi) dPhi-=2.0*pi;
        
        return dPhi;
    }
    

    double deltaEta(double eta1, double eta2){
        double dEta = (eta1-eta2);
        return dEta;
    }

    double deltaR(double eta1, double eta2, double phi1, double phi2) {
        double dEta = deltaEta(eta1, eta2);
        double dPhi = deltaPhi(phi1, phi2);
        return sqrt(dEta*dEta+dPhi*dPhi);
    }

};



template<typename FECODEC, typename DATA>
void HGCClusterAlgo<FECODEC,DATA>::run(const l1t::HGCFETriggerDigiCollection & coll, 
                                       const edm::EventSetup & es,
                                       const edm::Event & evt ) 
{

    es.get<IdealGeometryRecord>().get(HGCalEESensitive_, hgceeTopoHandle_);
    es.get<IdealGeometryRecord>().get(HGCalHESiliconSensitive_, hgchefTopoHandle_);


    for( const auto& digi : coll ) 
    {
        HGCalDetId module_id(digi.id());
        DATA data;
        data.reset();
        digi.decode(codec_, data);
        for(const auto& triggercell : data.payload)
        {
            if(triggercell.hwPt()>0)
            {


                HGCalDetId detid(triggercell.detId());
                int subdet = detid.subdetId();
                int cellThickness = 0;

                if( subdet == HGCEE ){ 
                    cellThickness = (hgceeTopoHandle_)->dddConstants().waferTypeL((unsigned int)detid.wafer() );
                }else if( subdet == HGCHEF ){
                    cellThickness = (hgchefTopoHandle_)->dddConstants().waferTypeL((unsigned int)detid.wafer() );
                }else if( subdet == HGCHEB ){
                    edm::LogWarning("DataNotFound") << "ATTENTION: the BH trgCells are not yet implemented !! ";
                }
                l1t::HGCalTriggerCell calibratedtriggercell( triggercell );
                calibration_.calibrate( calibratedtriggercell, cellThickness );     
                trgcell_product_->push_back( 0, calibratedtriggercell );
            }
        }
    }
    clustering_.clusterise( *trgcell_product_,  *cluster_product_ );
    multiclustering_.clusterise( *cluster_product_, *multicluster_product_ );
}// end run() 


typedef HGCClusterAlgo<HGCalTriggerCellBestChoiceCodec, HGCalTriggerCellBestChoiceCodec::data_type> HGCClusterAlgoBestChoice;
typedef HGCClusterAlgo<HGCalTriggerCellThresholdCodec, HGCalTriggerCellThresholdCodec::data_type> HGCClusterAlgoThreshold;

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
                  HGCClusterAlgoBestChoice,
                  "HGCClusterAlgoBestChoice");

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
                  HGCClusterAlgoThreshold,
                  "HGCClusterAlgoThreshold");
