# TruthInfo prototype

This package contains a prototype truth graph representation for CMSSW. The goal is to provide a compact, navigable, physics-oriented abstraction of the generator and simulation truth history of an event.

The current implementation is split into two layers:

1. `TruthGraph`: a compact raw graph built directly from existing CMS truth products.
2. `truth::Graph`: a higher-level logical graph exposing particles, vertices, payload, and navigation methods.

The prototype is intended for validation, reconstruction studies, visualization, and future truth-reco association work.

## Motivation

CMS currently exposes truth information through several low-level collections, such as HepMC, GenParticles, SimTracks, SimVertices, TrackingParticles, SimClusters, and CaloParticles. These collections are useful, but they encode different views of the event history and are often tied to detector-specific or production-specific conventions.

This package explores a different model: a single event-level truth graph that can be navigated using physics concepts.

Typical questions this should make easier are:

* Do two reconstructed objects come from the same parent particle?
* Did a given resonance, such as a Z boson, exist in the event history?
* Do two reconstructed objects come from the same Z boson?
* Which parton initiated a reconstructed jet?
* Is an object associated with the hard interaction or with pileup?
* Which detector-level interactions contributed to a reconstructed object?
* Should a reconstructed object be associated to a single truth particle, to a branch, or to an aggregated subgraph?

The intended user-facing API should allow reconstruction and validation code to operate on stable physics abstractions rather than directly depending on the storage details of `GenParticle`, `SimTrack`, `GenVertex`, or `SimVertex`.

## Package layout

```text
PhysicsTools/TruthInfo/
  interface/
    TruthGraph.h
    Graph.h
  src/
    TruthGraph.cc
    Graph.cc
    classes.h
    classes_def.xml
  plugins/
    TruthGraphProducer.cc
    TruthGraphDumper.cc
    TruthLogicalGraphProducer.cc
    TruthLogicalGraphDumper.cc
    BuildFile.xml
  python/
    truthGraphProducer_cfi.py
    truthLogicalGraphDumper_cfi.py
  BuildFile.xml
```

## Raw graph: `TruthGraph`

`TruthGraph` is a compact, read-only graph representation of event truth information. It is designed as an intermediate event data product built from existing CMS collections.

### Node types

The raw graph supports the following node kinds:

```cpp
enum class NodeKind : uint8_t {
  GenEvent,
  GenVertex,
  GenParticle,
  SimVertex,
  SimTrack
};
```

Each node stores a `NodeRef`:

```cpp
struct NodeRef {
  NodeKind kind;
  int64_t key;
};
```

The meaning of `key` depends on the node kind:

* `GenEvent`: generator connected-component id.
* `GenVertex`: HepMC barcode or HepMC3 vertex id.
* `GenParticle`: HepMC barcode or HepMC3 particle id.
* `SimVertex`: index in the `SimVertexContainer`.
* `SimTrack`: `SimTrack::trackId()`.

### Edge types

The raw graph supports edge categories:

```cpp
enum class EdgeKind : uint8_t {
  Gen,
  Sim,
  GenToSim,
  SimToGen
};
```

At present:

* `Gen` edges describe the generator graph.
* `Sim` edges describe the simulation graph.
* `GenToSim` edges connect matched generator particles or vertices to simulation nodes.
* `SimToGen` is reserved and is not produced by the current implementation.

### Storage model

Edges are stored in compressed sparse row form:

```cpp
std::vector<uint32_t> offsets;
std::vector<uint32_t> edges;
std::vector<uint8_t> edgeKind;
```

For node `i`, outgoing edges are stored in:

```cpp
edges[offsets[i] ... offsets[i + 1])
```

The corresponding edge kinds are stored in the same range of `edgeKind`.

The class provides convenience accessors such as:

```cpp
uint32_t nNodes() const;
uint32_t nEdges() const;
std::span<const uint32_t> children(uint32_t nodeId) const;
std::span<const uint8_t> childrenEdgeKinds(uint32_t nodeId) const;
const NodeRef& nodeRef(uint32_t nodeId) const;
bool isConsistent() const;
```

### Cached metadata

The raw graph stores lightweight metadata arrays parallel to the node list:

```cpp
std::vector<int32_t> pdgId;
std::vector<int16_t> status;
std::vector<uint64_t> eventId;
std::vector<int32_t> genEventOfNode;
```

It also stores association arrays:

```cpp
std::vector<int32_t> simTrackToGen;
std::vector<int32_t> simTrackToVtx;
```

These are indexed by raw node id. Entries that are not meaningful for a given node type are filled with default values, typically `0` or `-1`.

## `TruthGraphProducer`

`TruthGraphProducer` builds the raw `TruthGraph` from:

* HepMC3, when available;
* HepMC2, as fallback;
* `SimTrackContainer`;
* `SimVertexContainer`.

Default input tags are:

```python
genEventHepMC3 = cms.InputTag("generatorSmeared")
genEventHepMC  = cms.InputTag("generatorSmeared")
simTracks      = cms.InputTag("g4SimHits")
simVertices    = cms.InputTag("g4SimHits")
```

The producer creates:

* one `GenEvent` node per connected generator component;
* one node per generator vertex;
* one node per generator particle;
* one node per simulation vertex;
* one node per simulation track.

Generator components are computed using a disjoint-set union over the generator particle-vertex graph.

Simulation topology is built from:

* `SimTrack::vertIndex()` for `SimVertex -> SimTrack` edges;
* `SimVertex::parentIndex()` for `SimTrack -> SimVertex` edges.

Generator-to-simulation particle associations are built using `SimTrack::genpartIndex()` when available.

When enabled, cross-domain edges are also added:

* `GenParticle -> SimTrack`;
* `GenVertex -> SimVertex`, using the production vertex of the associated generator particle.

The cross edges are controlled by:

```python
addGenToSimEdges = cms.bool(True)
```

## Logical graph: `truth::Graph`

`truth::Graph` is the user-facing abstraction built from the raw `TruthGraph`. It is intended to expose a stable, physics-oriented API.

The logical graph is bipartite:

```text
Particle -> decay Vertex -> outgoing Particle
Particle <- production Vertex <- incoming Particle
```

The main public types are:

```cpp
namespace truth {
  class Graph;
  class Particle;
  class Vertex;

  struct ParticleData;
  struct VertexData;
  struct Checkpoint;
}
```

### `truth::Particle`

A `truth::Particle` may combine generator-level and simulation-level information when a robust correspondence exists.

The stored payload is:

```cpp
struct ParticleData {
  int32_t genNode = -1;
  int32_t simNode = -1;

  int32_t pdgId = 0;
  int16_t status = 0;
  uint64_t eventId = 0;
  int32_t genEvent = -1;

  math::XYZTLorentzVectorD momentum;
  std::vector<Checkpoint> checkpoints;
};
```

The convention for the momentum is "best available":

1. prefer simulation momentum when a matched `SimTrack` exists;
2. otherwise use generator momentum;
3. otherwise keep the default value.

Useful methods include:

```cpp
bool hasGen() const;
bool hasSim() const;
int32_t pdgId() const;
int16_t status() const;
uint64_t eventId() const;
int32_t genEvent() const;
const math::XYZTLorentzVectorD& momentum() const;

bool isRoot() const;
bool isLeaf() const;

std::vector<truth::Vertex> productionVertices() const;
std::vector<truth::Vertex> decayVertices() const;

std::vector<truth::Particle> parents() const;
std::vector<truth::Particle> children() const;
std::vector<truth::Particle> ancestors() const;
std::vector<truth::Particle> descendants() const;

bool hasAncestorPdgId(int pdgId) const;
std::optional<truth::Particle> firstAncestorWithPdgId(int pdgId) const;
std::optional<truth::Particle> firstCommonAncestor(truth::Particle other) const;
```

### `truth::Vertex`

A `truth::Vertex` stores vertex-level payload:

```cpp
struct VertexData {
  int32_t genNode = -1;
  int32_t simNode = -1;

  uint64_t eventId = 0;
  int32_t genEvent = -1;

  math::XYZTLorentzVectorD position;
};
```

Useful methods include:

```cpp
bool hasGen() const;
bool hasSim() const;
uint64_t eventId() const;
int32_t genEvent() const;
const math::XYZTLorentzVectorD& position() const;

bool isSource() const;
bool isSink() const;

std::vector<truth::Particle> incomingParticles() const;
std::vector<truth::Particle> outgoingParticles() const;
```

### Vertex treatment

At present, generator-level and simulation-level particles may be merged when a robust association exists.

Generator-level and simulation-level vertices are intentionally not merged. Each raw `GenVertex` and each raw `SimVertex` becomes a separate logical `truth::Vertex`.

This is deliberate for the current debugging phase. Keeping vertices separate makes it easier to diagnose:

* incorrect GEN-SIM associations;
* unexpected topologies;
* boundary-crossing behaviour;
* differences between generator and simulation vertex semantics.

Vertex merging can be revisited once the raw graph construction and association policy are better understood.

## Trajectory checkpoints

The logical particle model supports trajectory checkpoints:

```cpp
struct Checkpoint {
  uint32_t checkpointId = 0;
  math::XYZTLorentzVectorF position;
  math::XYZTLorentzVectorF momentum;
};
```

A checkpoint represents a relevant point along the propagation history of a particle.

The current prototype uses checkpoint `0` to store boundary-crossing information from `SimTrack`:

* position at boundary;
* momentum at boundary;
* boundary identifier.

This is meant to be a generic mechanism. A natural long-term direction is to build the truth graph directly while Geant4 tracks are being created and simulated, rather than reconstructing it afterwards from final CMS products. That would allow the graph to record multiple checkpoints along the propagation history and preserve information that is difficult to recover a posteriori.

## `TruthLogicalGraphProducer`

`TruthLogicalGraphProducer` builds a standalone `truth::Graph` from the raw `TruthGraph`.

Default input tags are:

```python
src            = cms.InputTag("truthGraphProducer")
simTracks      = cms.InputTag("g4SimHits")
simVertices    = cms.InputTag("g4SimHits")
genEventHepMC3 = cms.InputTag("generatorSmeared")
genEventHepMC  = cms.InputTag("generatorSmeared")
```

The producer performs the following steps:

1. Read and validate the raw `TruthGraph`.
2. Load optional payload from HepMC2, HepMC3, `SimTrackContainer`, and `SimVertexContainer`.
3. Assign temporary logical ids to raw particle and vertex nodes.
4. Merge only particle nodes across GEN and SIM when a robust association exists.
5. Keep GEN and SIM vertices separate.
6. Fill standalone `ParticleData` and `VertexData` payload.
7. Rebuild the logical bipartite graph in CSR form.
8. Validate the produced `truth::Graph`.

The produced graph is independent of the raw graph for ordinary navigation, but it keeps optional back-references to raw node ids for debugging:

```cpp
ParticleData::genNode
ParticleData::simNode
VertexData::genNode
VertexData::simNode
```

## Graph navigation examples

### Iterate over particles

```cpp
auto const& graph = event.get(truthGraphToken_);

for (auto particle : graph.particleViews()) {
  if (!particle.valid())
    continue;

  const auto pdgId = particle.pdgId();
  const auto p4 = particle.momentum();
}
```

### Find particles from a Z boson

```cpp
for (auto particle : graph.particleViews()) {
  if (particle.hasAncestorPdgId(23)) {
    // This particle has a Z boson somewhere in its ancestor chain.
  }
}
```

### Find the first common ancestor of two particles

```cpp
auto p1 = graph.particle(firstId);
auto p2 = graph.particle(secondId);

auto common = p1.firstCommonAncestor(p2);
if (common.has_value()) {
  const int pdgId = common->pdgId();
}
```

### Access production and decay vertices

```cpp
for (auto particle : graph.particleViews()) {
  for (auto vertex : particle.productionVertices()) {
    const auto x4 = vertex.position();
  }

  for (auto vertex : particle.decayVertices()) {
    const auto x4 = vertex.position();
  }
}
```

### Navigate parent and child particles

```cpp
for (auto particle : graph.particleViews()) {
  auto parents = particle.parents();
  auto children = particle.children();
}
```

### Access checkpoints

```cpp
for (auto particle : graph.particleViews()) {
  for (auto const& checkpoint : particle.checkpoints()) {
    const auto id = checkpoint.checkpointId;
    const auto position = checkpoint.position;
    const auto momentum = checkpoint.momentum;
  }
}
```

## Dumping and visualization

Two dumper modules are provided.

### Raw graph dumper

`TruthGraphDumper` writes a DOT representation of the raw graph.

It includes enriched labels using HepMC and simulation products when available.

Default output:

```text
truthgraph.dot
```

The dumper can be configured with:

```python
src = cms.InputTag("truthGraphProducer")
dotFile = cms.string("truthgraph.dot")
maxNodes = cms.uint32(5000)
maxEdgesPerNode = cms.uint32(200)
```

### Logical graph dumper

`TruthLogicalGraphDumper` writes a DOT representation of the logical graph.

Default output:

```text
truthlogicalgraph.dot
```

The dumper can also use the raw graph to enrich labels with raw node information:

```python
truthLogicalGraphDumper = cms.EDAnalyzer(
    "TruthLogicalGraphDumper",
    src = cms.InputTag("truthLogicalGraphProducer"),
    rawSrc = cms.InputTag("truthGraphProducer"),
    dotFile = cms.string("truthlogicalgraph.dot"),
    maxParticles = cms.uint32(5000),
    maxVertices = cms.uint32(5000),
    maxEdgesPerNode = cms.uint32(200),
)
```

DOT files can be converted with Graphviz, for example:

```bash
dot -Tpdf truthlogicalgraph.dot -o truthlogicalgraph.pdf
```

or:

```bash
dot -Tpng truthlogicalgraph.dot -o truthlogicalgraph.png
```

## Example configuration

A minimal test path can be configured as follows:

```python
import FWCore.ParameterSet.Config as cms

process.load("PhysicsTools.TruthInfo.truthGraphProducer_cfi")

process.truthLogicalGraphProducer = cms.EDProducer(
    "TruthLogicalGraphProducer",
    src = cms.InputTag("truthGraphProducer"),
    simTracks = cms.InputTag("g4SimHits"),
    simVertices = cms.InputTag("g4SimHits"),
    genEventHepMC3 = cms.InputTag("generatorSmeared"),
    genEventHepMC = cms.InputTag("generatorSmeared"),
)

process.truthLogicalGraphDumper = cms.EDAnalyzer(
    "TruthLogicalGraphDumper",
    src = cms.InputTag("truthLogicalGraphProducer"),
    rawSrc = cms.InputTag("truthGraphProducer"),
    dotFile = cms.string("truthlogicalgraph.dot"),
    maxParticles = cms.uint32(5000),
    maxVertices = cms.uint32(5000),
    maxEdgesPerNode = cms.uint32(200),
)

process.p = cms.Path(
    process.truthGraphProducer +
    process.truthLogicalGraphProducer +
    process.truthLogicalGraphDumper
)
```

The raw graph producer can be customized explicitly if needed:

```python
process.truthGraphProducer = cms.EDProducer(
    "TruthGraphProducer",
    genEventHepMC3 = cms.InputTag("generatorSmeared"),
    genEventHepMC = cms.InputTag("generatorSmeared"),
    simTracks = cms.InputTag("g4SimHits"),
    simVertices = cms.InputTag("g4SimHits"),
    addGenToSimEdges = cms.bool(True),
)
```

## Intended matching strategy

The long-term matching model is to build detector-level associators from hits to `truth::Particle`.

In this model:

* the graph provides the particle and vertex structure;
* detector-level truth association is performed through hits;
* reconstructed objects can be associated either to a single `truth::Particle` or to a larger `truth::Branch`;
* truth information can be aggregated over a coherent branch when a reconstructed object corresponds to a composite truth structure.

Different reconstruction domains need different matching metrics:

* tracking association is usually based on shared hits, not on hit energy;
* calorimeter clustering association can use energy fractions or energy-weighted metrics;
* other reconstruction objects may require detector-specific metrics.

The graph should therefore act as the common truth substrate, while matching definitions remain detector-aware and use-case dependent.

## Possible future `truth::Branch` abstraction

A future `truth::Branch` abstraction is intended to represent a coherent subgraph selected from the full truth graph.

This would provide a natural target for truth-reco association when a reconstructed object is not well described by a single truth particle.

Possible branch-level operations include:

* aggregate particles in a physically connected subgraph;
* collect all detector hits attached to particles in the branch;
* compute detector-specific matching scores;
* compare two branches;
* define stable references for composite truth structures.

In this picture, `truth::Branch` avoids forcing an early collapse of the truth history into fixed reference objects.

## Hits attached to particles

A possible long-term direction is to attach detector hits of different types directly to `truth::Particle`, or to provide hit collections indexed by `truth::Particle` ids.

This would make the graph the central truth data structure. Detector-specific truth objects such as `TrackingParticle`, `SimCluster`, and `CaloParticle` could then be expressed as views or derived abstractions on top of the same graph.

This would make it possible to:

* use one common truth structure across subsystems;
* preserve detector-specific information without fragmenting the truth model;
* define multiple matching strategies on top of the same graph;
* aggregate truth information dynamically over particles, vertices, or branches.

## Current status

The current prototype can:

* build a raw `TruthGraph` from generator and simulation products;
* build a logical `truth::Graph` from the raw graph;
* merge matched generator and simulation particles;
* keep generator and simulation vertices separate for debugging;
* navigate particle and vertex relations;
* compute parents, children, ancestors, descendants, roots, and leaves;
* find ancestors with a given PDG id;
* find a common ancestor between two particles;
* store boundary-crossing checkpoints;
* dump raw and logical graphs to DOT for debugging.

## Known limitations

The current implementation is a prototype and several aspects are intentionally conservative.

Known limitations include:

* vertex merging is not yet attempted;
* only a limited set of GEN-SIM associations is currently used;
* checkpoint information is limited to boundary-crossing information from `SimTrack`;
* hit-to-particle association is not yet implemented;
* `truth::Branch` is part of the target design but not yet implemented;
* the logical API is still evolving;
* the raw graph construction needs further validation on realistic events and pileup scenarios.

## Next steps

Planned or natural next steps are:

1. Continue debugging the raw truth graph construction.
2. Validate HepMC2 and HepMC3 behaviour on representative workflows.
3. Refine GEN-SIM particle association.
4. Study whether, when, and how generator and simulation vertices should be merged.
5. Extend the checkpoint model beyond boundary crossings.
6. Investigate direct graph construction during Geant4 simulation.
7. Define hit-to-`truth::Particle` associators.
8. Prototype detector-specific matching metrics on top of the graph.
9. Implement a `truth::Branch` abstraction.
10. Study how existing detector-specific truth containers could be represented as views over the graph.
11. Stabilize the logical API for downstream reconstruction and validation code.

## Design principle

The main design principle is to separate storage details from physics navigation.

Low-level CMS products remain the source of truth for building the graph, but downstream code should be able to ask physics questions through a stable interface:

```cpp
particle.parents();
particle.children();
particle.ancestors();
particle.firstCommonAncestor(other);
particle.hasAncestorPdgId(23);
```

rather than reimplementing event-history navigation separately for each reconstruction or validation use case.
