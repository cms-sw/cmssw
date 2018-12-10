#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalClusteringImpl.h"


class HGCalBackendLayer1Processor2DClustering : public HGCalBackendLayer1ProcessorBase 
{    
  public:
    HGCalBackendLayer1Processor2DClustering(const edm::ParameterSet& conf) :
      HGCalBackendLayer1ProcessorBase(conf),
      clustering_( conf.getParameterSet("C2d_parameters") )
    {
      std::string typeCluster(conf.getParameterSet("C2d_parameters").getParameter<std::string>("clusterType"));
      if(typeCluster=="dRC2d"){
        clusteringAlgorithmType_ = dRC2d;
      }else if(typeCluster=="NNC2d"){
        clusteringAlgorithmType_ = NNC2d;
      }else if(typeCluster=="dRNNC2d"){
        clusteringAlgorithmType_ = dRNNC2d;
      }else {
        throw cms::Exception("HGCTriggerParameterError")
                           << "Unknown clustering type '" << typeCluster;
      }
    }

    void run(const edm::Handle<l1t::HGCalTriggerCellBxCollection>& collHandle,
             l1t::HGCalClusterBxCollection& collCluster2D,
             const edm::EventSetup& es) override 
    {
      es.get<CaloGeometryRecord>().get("", triggerGeometry_);
      clustering_.eventSetup(es);

      /* create a persistent vector of pointers to the trigger-cells */
      std::vector<edm::Ptr<l1t::HGCalTriggerCell>> triggerCellsPtrs;
      for( unsigned i = 0; i < collHandle->size(); ++i ) {
        edm::Ptr<l1t::HGCalTriggerCell> ptr(collHandle,i);
        triggerCellsPtrs.push_back(ptr);
      }

      std::sort(triggerCellsPtrs.begin(), triggerCellsPtrs.end(),
           [](const edm::Ptr<l1t::HGCalTriggerCell>& a, 
              const  edm::Ptr<l1t::HGCalTriggerCell>& b) -> bool
            {
              return a->mipPt() > b->mipPt();
            }
          );

      /* call to C2d clustering */
      switch(clusteringAlgorithmType_){
        case dRC2d : 
          clustering_.clusterizeDR(triggerCellsPtrs, collCluster2D);
          break;
        case NNC2d:
          clustering_.clusterizeNN( triggerCellsPtrs, collCluster2D, *triggerGeometry_ );
          break;
        case dRNNC2d:
          clustering_.clusterizeDRNN( triggerCellsPtrs, collCluster2D, *triggerGeometry_ );
          break;
        default:
          // Should not happen, clustering type checked in constructor
          break;
      }
    }  
    
  private:
    enum ClusterType{
      dRC2d,
      NNC2d,
      dRNNC2d
    };

    edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

    /* algorithms instances */
    HGCalClusteringImpl clustering_;

    /* algorithm type */
    ClusterType clusteringAlgorithmType_;
};

DEFINE_EDM_PLUGIN(HGCalBackendLayer1Factory, 
                  HGCalBackendLayer1Processor2DClustering,
                  "HGCalBackendLayer1Processor2DClustering");
