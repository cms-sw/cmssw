#include <iostream>
#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoAClustersDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoAClustersFilteredMaskDeviceCollection.h"
#include "RecoHGCal/TICL/plugins/alpaka/ClusterFilterSoAByAlgoAndSize.h"
#include "DataFormats/Portable/interface/alpaka/PortableObject.h"

#include "DataFormats/Common/interface/StdArray.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    class FilteredLayerClustersSoAProducer : public stream::EDProducer<> {
        public:
            FilteredLayerClustersSoAProducer(edm::ParameterSet const& config)
            :   min_cluster_size_(config.getParameter<int>("min_cluster_size")),
                max_cluster_size_(config.getParameter<int>("max_cluster_size")),
                clusters_token_(consumes(config.getParameter<edm::InputTag>("LayerClustersSoA"))),
                clusters_mask_token_{produces()}
            {}
            
            ~FilteredLayerClustersSoAProducer() override = default;

            static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
                edm::ParameterSetDescription desc;
                desc.add<edm::InputTag>("LayerClustersSoA", edm::InputTag("hltHgcalSoALayerClustersProducer"));
                desc.add<int>("min_cluster_size", 0);
                desc.add<int>("max_cluster_size", 9999);
                descriptions.addWithDefaultLabel(desc);
            }

            void produce(device::Event& evt, device::EventSetup const& es) override {
                std::cout<< "FilteredLayerClustersSoAProducer";
                auto const& layerClustersSoA = evt.get(clusters_token_);
                auto const layerClustersSoAConstView = layerClustersSoA.view();
               
                HGCalSoAClustersFilteredMaskDeviceCollection output(layerClustersSoAConstView.metadata().size(), evt.queue());
                auto outputView = output.view();

                theFilter_->filter(evt.queue(), layerClustersSoAConstView, outputView, min_cluster_size_, max_cluster_size_);
                evt.emplace(clusters_mask_token_, std::move(output));
            }

        private:
            const int min_cluster_size_;
            const int max_cluster_size_;
            device::EDGetToken<HGCalSoAClustersDeviceCollection> const clusters_token_;
            device::EDPutToken<HGCalSoAClustersFilteredMaskDeviceCollection> const clusters_mask_token_;
            std::unique_ptr<ClusterFilterSoAByAlgoAndSize> theFilter_;
    };

} // end ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(FilteredLayerClustersSoAProducer);