#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithmFactory.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "FWCore/Utilities/interface/FunctionWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
//#include "CommonTools/Utils/src/MethodInvoker.h"
//#include "CommonTools/Utils/src/AnyMethodArgument.h"

#include "Reflex/Type.h"
#include "Reflex/Member.h"
#include "Reflex/Object.h"

#include "TDataType.h"

#include <string>
#include <iostream>

class TrackingRecHitNoSmearingPlugin:
    public TrackingRecHitAlgorithm
{
    public:
        TrackingRecHitNoSmearingPlugin(
            const std::string& name,
            const edm::ParameterSet& config,
            edm::ConsumesCollector& consumesCollector
        )
        {
            std::cout<<"created plugin with name: "<<name<<std::endl;
        }

        virtual std::vector<SiTrackerGSRecHit2D> processDetId(
            const DetId& detId,
            const TrackerTopology& trackerTopology,
            const TrackerGeometry& trackerGeometry,
            const std::vector<const PSimHit*>& simHits
        ) const
        {
            /*
            DetId id(detId);
            edm::ObjectWithDict obj(typeid(detId),&id);
            edm::TypeWithDict oType(typeid(detId));
            edm::FunctionWithDict fct = oType.functionMemberByName("subdetId");
            unsigned int r = 0;
            edm::ObjectWithDict ret(typeid(unsigned int),&r);
            fct.invoke(obj,&ret);
            std::cout<<"subdet: "<<r<<std::endl;
            */
            /*
            DetId id(detId);
            TrackerTopology tTopol(trackerTopology);

            std::cout<<tTopol.pxbLadder(id)<<std::endl;
            edm::ObjectWithDict obj_detId(typeid(detId),&id);
            edm::ObjectWithDict obj_trackerTopology(typeid(tTopol),&tTopol);

            edm::TypeWithDict oType(typeid(trackerTopology));
            edm::FunctionWithDict fct = oType.functionMemberByName("pxbLadder");
            std::cout<<fct.size()<<std::endl;
            unsigned int r = 0;
            edm::ObjectWithDict ret(typeid(unsigned int),&r);
            std::vector<void*> args;
            args.push_back(&obj_detId);

            //fct.invoke(obj_trackerTopology,&ret);

            std::cout<<"layer: "<<r<<std::endl;
        `   */

            /*
            DetId id(detId);
            TrackerTopology tTopol(trackerTopology);
            Reflex::Type t = Reflex::Type::ByTypeInfo(typeid(tTopol));
            TDataType* dataType = TDataType::GetDataType(TDataType::GetType(typeid(tTopol)));
            (void)dataType;
            std::cout<<t.Name_c_str () <<std::endl;
            for (Reflex::Member_Iterator mi = t.Member_Begin(); mi != t.Member_End(); ++mi)
            {
              switch ((*mi).MemberType())
              {
                case Reflex::DATAMEMBER     : std::cout << "Datamember: " << (*mi).Name() << " at offset " << (*mi).Offset() << std::endl;
                case Reflex::FUNCTIONMEMBER : std::cout << "Functionmember: " << (*mi).Name() << " has " << (*mi).FunctionParameterSize() << " parameters " << std::endl;
                default             : std::cout << "This should never happen" << std::endl;
              }
            }
            */
/*
            SiTrackerGSRecHit2D* recHit = new SiTrackerGSRecHit2D(
                  position,
                  error,
                  theDetUnit,
                  simHitCounter,
                  trackID,
                  eeID,
                  ClusterRef(FastTrackerClusterRefProd, simHitCounter),
                  alphaMult,
                  betaMult
              );
              */
            std::vector<SiTrackerGSRecHit2D> recHits;
            return recHits;
        }

};

DEFINE_EDM_PLUGIN(
    TrackingRecHitAlgorithmFactory,
    TrackingRecHitNoSmearingPlugin,
    "TrackingRecHitNoSmearingPlugin"
);

