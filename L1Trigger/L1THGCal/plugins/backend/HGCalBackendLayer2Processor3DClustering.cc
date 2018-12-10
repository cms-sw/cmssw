#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalMulticlusteringImpl.h"    


class HGCalBackendLayer2Processor3DClustering : public HGCalBackendLayer2ProcessorBase 
{
  public:
    HGCalBackendLayer2Processor3DClustering(const edm::ParameterSet& conf)  : 
      HGCalBackendLayer2ProcessorBase(conf),
      multiclustering_( conf.getParameterSet("C3d_parameters") )
    {
      std::string typeMulticluster(conf.getParameterSet("C3d_parameters").getParameter<std::string>("type_multicluster"));
      if(typeMulticluster=="dRC3d"){
        multiclusteringAlgoType_ = dRC3d;
      }else if(typeMulticluster=="DBSCANC3d"){
        multiclusteringAlgoType_ = DBSCANC3d;
      }else {
        throw cms::Exception("HGCTriggerParameterError")
          << "Unknown Multiclustering type '" << typeMulticluster;
      }
    }
        
    void run(const edm::Handle<l1t::HGCalClusterBxCollection>& collHandle,
             l1t::HGCalMulticlusterBxCollection& collCluster3D,
             const edm::EventSetup& es) override 
    {
      es.get<CaloGeometryRecord>().get("", triggerGeometry_);
      multiclustering_.eventSetup(es);

      /* create a persistent vector of pointers to the trigger-cells */
      std::vector<edm::Ptr<l1t::HGCalCluster>> clustersPtrs;
      for( unsigned i = 0; i < collHandle->size(); ++i ) {
      edm::Ptr<l1t::HGCalCluster> ptr(collHandle,i);
        clustersPtrs.push_back(ptr);
      }

      /* call to multiclustering and compute shower shape*/
      switch(multiclusteringAlgoType_){
        case dRC3d : 
          multiclustering_.clusterizeDR( clustersPtrs, collCluster3D, *triggerGeometry_);
          break;
        case DBSCANC3d:
          multiclustering_.clusterizeDBSCAN( clustersPtrs, collCluster3D, *triggerGeometry_);
          break;
        default:
          // Should not happen, clustering type checked in constructor
          break;
      }
    }
    
  private:
    enum MulticlusterType{
      dRC3d,
      DBSCANC3d
    };
        
    edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

    /* algorithms instances */
    HGCalMulticlusteringImpl multiclustering_;

    /* algorithm type */
    MulticlusterType multiclusteringAlgoType_;
};

DEFINE_EDM_PLUGIN(HGCalBackendLayer2Factory, 
                  HGCalBackendLayer2Processor3DClustering,
                  "HGCalBackendLayer2Processor3DClustering");
