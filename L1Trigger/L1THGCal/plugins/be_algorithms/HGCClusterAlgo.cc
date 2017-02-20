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
        HGCClusterAlgo(const edm::ParameterSet& conf, edm::ConsumesCollector& cc):
            Algorithm<FECODEC>(conf,cc),
            trgcell_product_( new l1t::HGCalTriggerCellBxCollection),
            cluster_product_( new l1t::HGCalClusterBxCollection ),
            multicluster_product_( new l1t::HGCalMulticlusterBxCollection ),
            HGCalEESensitive_(conf.getParameter<std::string>("HGCalEESensitive_tag")),
            HGCalHESiliconSensitive_(conf.getParameter<std::string>("HGCalHESiliconSensitive_tag")),
            calibration_(conf.getParameterSet("calib_parameters")),
            clustering_(conf.getParameterSet("C2d_parameters")),
            multiclustering_(conf.getParameterSet("C3d_parameters")){}

        typedef std::unique_ptr<HGCalTriggerGeometryBase> ReturnType;

        virtual void setProduces(edm::EDProducer& prod) const override final 
        {
            prod.produces<l1t::HGCalClusterBxCollection>(name());
            prod.produces<l1t::HGCalMulticlusterBxCollection>("cluster3D");

        }

        virtual void run(const l1t::HGCFETriggerDigiCollection& coll, const edm::EventSetup& es, const edm::Event&evt ) override final;
        virtual void putInEvent(edm::Event& evt) override final 
        {
            //evt.put(std::move(trgcell_product_),name());
            evt.put(std::move(cluster_product_),name());
            evt.put(std::move(multicluster_product_),"cluster3D");
        }

        virtual void reset() override final 
        {
            trgcell_product_.reset( new l1t::HGCalTriggerCellBxCollection);
            cluster_product_.reset( new l1t::HGCalClusterBxCollection );            
            multicluster_product_.reset( new l1t::HGCalMulticlusterBxCollection );            
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


/*****************************************************************/
template<typename FECODEC, typename DATA>
void HGCClusterAlgo<FECODEC,DATA>::run(const l1t::HGCFETriggerDigiCollection& coll, 
                                       const edm::EventSetup& es,
                                       const edm::Event&evt
    ) 
/*****************************************************************/
{
    //virtual void run(const l1t::HGCFETriggerDigiCollection& coll, const edm::EventSetup& es, const edm::Event&evt) override final
    //   {
            es.get<IdealGeometryRecord>().get(HGCalEESensitive_, hgceeTopoHandle_);
            es.get<IdealGeometryRecord>().get(HGCalHESiliconSensitive_, hgchefTopoHandle_);
            std::cout << "CLUSTERING PARAMETERS: "<< std::endl;
            std::cout << "------ Clustering thresholds for trigger cells to be included in C2d: " << tc_CUT_ << std::endl;
            std::cout << "------ Seeding thresholds to start the clusterization procedure: " << seed_CUT_ << std::endl; 
            std::cout << "------ Max-Distance in the normalized plane to search for next-layer C2d to merge into the C3d: " << dR_forC3d_ << std::endl;  
            
            std::cout << "coll size " << coll.size() << std::endl;
            
//==================================== Get The calibrated trigger cell collection

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
                        l1t::HGCalTriggerCell calibratedtriggercell(triggercell);
                        calibration_.calibrate(calibratedtriggercell, cellThickness);     
                        trgcell_product_->push_back(0,calibratedtriggercell);
                    }
                }
            }
        


//====================================

//:uca
//:uca
//:uca
//:uca
//:uca
//std::unique_ptr<l1t::HGCalClusterBxCollection> 
            clustering_.clusterizeBase( trgcell_product_,  cluster_product_ );
            multiclustering_.clusterizeMultiple( cluster_product_, multicluster_product_ );
//:uca
//:uca//CODE THE REAL C3D-ALGORITHM HERE: using previous built C2D  + fill all information transmittable to the CORRELATOR                
//:uca
//if(cluster_product_->size()>0){
//:uca    std::vector<size_t> isMerged;
//:uca
//:uca    size_t seedx=0;
//:uca    for(l1t::HGCalClusterBxCollection::const_iterator c2d = cluster_product_->begin(); c2d != cluster_product_->end(); ++c2d, ++seedx){
//:uca        l1t::HGCalMulticluster cluster3D( reco::LeafCandidate::LorentzVector(), 0, 0, 0);
//:uca        double_t tmpEta = 0.;
//:uca        double_t tmpPhi = 0.;           
//:uca        double_t C3d_pt  = 0.;
//:uca        double_t C3d_eta = 0.;
//:uca        double_t C3d_phi = 0.;
//:uca        uint32_t C3d_hwPtEm = 0;
//:uca        uint32_t C3d_hwPtHad = 0;
//:uca        uint32_t totLayer = 0;
//:uca
//:uca        bool skip=false;
//:uca        
//:uca        //std::cout << "In the C2d collection, seed the C3d with this : " << seedx << " - "<< c2d->p4().Pt() << " eta: " <<  c2d->p4().Eta() << " --> layer " << c2d->layer() << "  skip before 2nd loop "<< skip << std::endl;                
//:uca        
//:uca        size_t idx=0;
//:uca        for(l1t::HGCalClusterBxCollection::const_iterator c2d_aux = cluster_product_->begin(); c2d_aux != cluster_product_->end(); ++c2d_aux, ++idx){
//:uca            //  std::cout << "     loop over C2d again and search for match:" << "   idx: " << idx << "  eta: " << c2d_aux->p4().Eta() << std::endl;
//:uca            //std::cout << "   before isMerged loop: " << skip<< std::endl;
//:uca            for(size_t i(0); i<isMerged.size(); i++){
//:uca                //std::cout <<  isMerged.at(i) << ", ";
//:uca                if(idx==isMerged.at(i)){
//:uca                    skip=true;
//:uca                    continue;
//:uca                }
//:uca            }
//:uca            //std::cout << "\n";
//:uca            double dR =  deltaR( c2d->p4().Eta(), c2d_aux->p4().Eta(), c2d->p4().Phi(), c2d_aux->p4().Phi() ); 
//:uca            std::cout << "looping on the c2d directly from the collection : "<< c2d->p4().pt() << "  --> layer " << c2d->layer() << " dR: " << dR << "  SKIP var = " << skip << std::endl;
//:uca            
//:uca            if(skip){
//:uca                skip=false;
//:uca                //std::cout << "     the c2d considered has been already merged!!";
//:uca                continue;
//:uca            }
//:uca            if( dR < 0.1 ){
//:uca                //    std::cout << "     The idx "<< idx << " C2d has been matched and kept for 3D to the " << seedx 
//:uca                //          << " - "<< c2d_aux->p4().Pt() << " eta: " <<  c2d_aux->p4().Eta() 
//:uca                //          << " --> layer " << c2d_aux->layer() << std::endl;             
//:uca                isMerged.push_back(idx);
//:uca                tmpEta+=c2d_aux->p4().Eta() * c2d_aux->p4().Pt();
//:uca                tmpPhi+=c2d_aux->p4().Phi() * c2d_aux->p4().Pt();
//:uca                C3d_pt+=c2d_aux->p4().Pt();
//:uca                C3d_hwPtEm+=c2d_aux->hwPtEm();
//:uca                C3d_hwPtHad+=c2d_aux->hwPtHad();
//:uca                totLayer++;
//:uca            }
//:uca        }
//:uca        
//:uca        std::cout <<"STO PER ENTRARE NEL MULTICLUSTERING" << std::endl;
//:uca        if( totLayer > 2){
//:uca            cluster3D.setNtotLayer(totLayer);
//:uca            cluster3D.setHwPtEm(C3d_hwPtEm);
//:uca            cluster3D.setHwPtHad(C3d_hwPtHad);
//:uca            C3d_eta=tmpEta/C3d_pt;
//:uca            C3d_phi=tmpPhi/C3d_pt;                
//:uca            math::PtEtaPhiMLorentzVector calib3dP4(C3d_pt, C3d_eta, C3d_phi, 0 );
//:uca            cluster3D.setP4(calib3dP4);                    
//:uca            std::cout << "  A MULTICLUSTER has been built with pt, eta, phi = " << C3d_pt << ", " << C3d_eta << ", "<< C3d_phi <<  std::endl;
//:uca            cluster3D_product_->push_back(0,cluster3D);
//:uca        }                    
//:uca    }
//:uca}
}



typedef HGCClusterAlgo<HGCalTriggerCellBestChoiceCodec, HGCalTriggerCellBestChoiceCodec::data_type> HGCClusterAlgoBestChoice;
typedef HGCClusterAlgo<HGCalTriggerCellThresholdCodec, HGCalTriggerCellThresholdCodec::data_type> HGCClusterAlgoThreshold;

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        HGCClusterAlgoBestChoice,
        "HGCClusterAlgoBestChoice");

DEFINE_EDM_PLUGIN(HGCalTriggerBackendAlgorithmFactory, 
        HGCClusterAlgoThreshold,
        "HGCClusterAlgoThreshold");
