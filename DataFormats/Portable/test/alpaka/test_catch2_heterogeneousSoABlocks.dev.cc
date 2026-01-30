#include <Eigen/Core>
#include <Eigen/Dense>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

// This test checks the correctness of using SoABlocks with PortableCollections.

GENERATE_SOA_LAYOUT(NodesT, SOA_COLUMN(int, id), SOA_SCALAR(int, count))

using Nodes = NodesT<>;

GENERATE_SOA_LAYOUT(EdgesT, SOA_COLUMN(int, src), SOA_COLUMN(int, dst), SOA_COLUMN(float, cost), SOA_SCALAR(int, count))

using Edges = EdgesT<>;

GENERATE_SOA_BLOCKS(OneBlockTemplate, SOA_BLOCK(nodes, NodesT))
GENERATE_SOA_BLOCKS(GraphT, SOA_BLOCK(nodes, NodesT), SOA_BLOCK(edges, EdgesT))

using OneBlock = OneBlockTemplate<>;
using OneBlockView = OneBlock::View;
using OneBlockConstView = OneBlock::ConstView;

using Graph = GraphT<>;
using GraphView = Graph::View;
using GraphConstView = Graph::ConstView;

// Fill SoAs
struct FillSoAs {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, Nodes::View nodes, Edges::View edges) const {
    const int N = static_cast<int>(nodes.metadata().size());
    const int E = static_cast<int>(edges.metadata().size());

    // Fill nodes with the indexes
    for (auto i : cms::alpakatools::uniform_elements(acc, nodes.metadata().size())) {
      nodes[i].id() = static_cast<int>(i);
    }
    if (cms::alpakatools::once_per_grid(acc)) {
      nodes.count() = N;
    }

    // Fill edges with some arbitrary but deterministic values
    for (auto j : cms::alpakatools::uniform_elements(acc, edges.metadata().size())) {
      int src = static_cast<int>(j % N);
      int dst = static_cast<int>((j * 7 + 3) % N);
      edges[j].src() = src;
      edges[j].dst() = dst;
      edges[j].cost() = 0.5f * float(src + dst);
    }
    if (cms::alpakatools::once_per_grid(acc)) {
      edges.count() = E;
    }
  }
};

// Fill one block SoABlocks
struct FillOneBlockSoABlocks {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, OneBlockView blocksView) const {
    const int N = static_cast<int>(blocksView.nodes().metadata().size());

    // Fill nodes with the indexes
    for (auto i : cms::alpakatools::uniform_elements(acc, blocksView.nodes().metadata().size())) {
      blocksView.nodes()[i].id() = static_cast<int>(i);
    }
    if (cms::alpakatools::once_per_grid(acc)) {
      blocksView.nodes().count() = N;
    }
  }
};

// Fill SoABlocks
struct FillBlocks {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, GraphView blocksView) const {
    const int N = static_cast<int>(blocksView.nodes().metadata().size());
    const int E = static_cast<int>(blocksView.edges().metadata().size());

    // Fill nodes with the indexes
    for (auto i : cms::alpakatools::uniform_elements(acc, blocksView.nodes().metadata().size())) {
      blocksView.nodes()[i].id() = static_cast<int>(i);
    }
    if (cms::alpakatools::once_per_grid(acc)) {
      blocksView.nodes().count() = N;
    }

    // Fill edges with some arbitrary but deterministic values
    for (auto j : cms::alpakatools::uniform_elements(acc, blocksView.edges().metadata().size())) {
      int src = static_cast<int>(j % N);
      int dst = static_cast<int>((j * 7 + 3) % N);
      blocksView.edges()[j].src() = src;
      blocksView.edges()[j].dst() = dst;
      blocksView.edges()[j].cost() = 0.5f * float(src + dst);
    }
    if (cms::alpakatools::once_per_grid(acc)) {
      blocksView.edges().count() = E;
    }
  }
};

TEST_CASE("SoABlocks minimal graph in heterogeneous environment") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cout << "No devices available for the " << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)
              << " backend, skipping.\n";
    return;
  }

  for (auto const& device : devices) {
    std::cout << "Running on " << alpaka::getName(device) << std::endl;
    Queue queue(device);

    // Number of elements
    const int N = 50;
    const int E = 120;

    // Portable Collections for SoAs
    PortableCollection<Device, Nodes> nodesCollection(queue, N);
    PortableCollection<Device, Edges> edgesCollection(queue, E);
    Nodes::View& nodesCollectionView = nodesCollection.view();
    Edges::View& edgesCollectionView = edgesCollection.view();

    // Portable Collection for SoABlocks
    PortableCollection<Device, OneBlock> oneBlockCollection(queue, N);
    OneBlockView& oneBlockCollectionView = oneBlockCollection.view();

    PortableCollection<Device, Graph> graphCollection(queue, N, E);
    GraphView& graphCollectionView = graphCollection.view();

    // Work division
    const std::size_t blockSize = 256;
    const std::size_t numberOfBlocksOneBlockVersion = cms::alpakatools::divide_up_by(N, blockSize);
    const auto workDivOneBlockVersion = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocksOneBlockVersion, blockSize);

    const std::size_t maxElems = std::max<std::size_t>(N, E);
    const std::size_t numberOfBlocks = cms::alpakatools::divide_up_by(maxElems, blockSize);
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    // Fill: separate e blocks
    alpaka::exec<Acc1D>(queue, workDiv, FillSoAs{}, nodesCollectionView, edgesCollectionView);
    alpaka::exec<Acc1D>(queue, workDivOneBlockVersion, FillOneBlockSoABlocks{}, oneBlockCollectionView);
    alpaka::exec<Acc1D>(queue, workDiv, FillBlocks{}, graphCollectionView);
    alpaka::wait(queue);

    // Check results on host
    PortableHostCollection<Nodes> nodesHost(cms::alpakatools::host(), N);
    PortableHostCollection<Edges> edgesHost(cms::alpakatools::host(), E);
    PortableHostCollection<OneBlock> oneBlockHost(cms::alpakatools::host(), N);
    PortableHostCollection<Graph> graphHost(cms::alpakatools::host(), N, E);

    alpaka::memcpy(queue, nodesHost.buffer(), nodesCollection.buffer());
    alpaka::memcpy(queue, edgesHost.buffer(), edgesCollection.buffer());
    alpaka::memcpy(queue, oneBlockHost.buffer(), oneBlockCollection.buffer());
    alpaka::memcpy(queue, graphHost.buffer(), graphCollection.buffer());
    alpaka::wait(queue);

    const Nodes::ConstView nodesHostView = nodesHost.const_view();
    const Edges::ConstView edgesHostView = edgesHost.const_view();
    const OneBlockConstView oneBlockHostView = oneBlockHost.const_view();
    const GraphConstView graphHostView = graphHost.const_view();

    // Nodes
    REQUIRE(graphHostView.nodes().count() == N);
    for (int i = 0; i < N; ++i) {
      REQUIRE(oneBlockHostView.nodes()[i].id() == nodesHostView[i].id());
      REQUIRE(oneBlockHostView.nodes()[i].id() == i);
      REQUIRE(graphHostView.nodes()[i].id() == nodesHostView[i].id());
      REQUIRE(graphHostView.nodes()[i].id() == i);
    }

    // Edges
    REQUIRE(graphHostView.edges().count() == E);
    for (int j = 0; j < E; ++j) {
      REQUIRE(graphHostView.edges()[j].src() == edgesHostView[j].src());
      REQUIRE(graphHostView.edges()[j].dst() == edgesHostView[j].dst());
      REQUIRE(graphHostView.edges()[j].cost() == edgesHostView[j].cost());

      int src = j % N;
      int dst = (j * 7 + 3) % N;
      REQUIRE(graphHostView.edges()[j].src() == src);
      REQUIRE(graphHostView.edges()[j].dst() == dst);
      REQUIRE_THAT(graphHostView.edges()[j].cost(), Catch::Matchers::WithinAbs(0.5f * float(src + dst), 1e-6));
    }
  }
}
