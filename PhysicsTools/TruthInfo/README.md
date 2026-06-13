# TruthInfo prototype

This package contains a prototype truth graph representation for CMSSW. The goal is to provide a compact, navigable, physics-oriented abstraction of the generator, simulation, and detector-hit truth history of an event.

The current implementation is split into three layers:

1. `TruthGraph`: a compact raw graph built directly from existing CMS truth products.
2. `truth::Graph`: a higher-level logical graph exposing particles, vertices, payload, and navigation methods.
3. `truth::LogicalGraphHitIndex`: an auxiliary hit index associating logical particles to calorimeter SimHits and, when available, to reconstructed RecHits.

The prototype is intended for validation, reconstruction studies, visualization, and future truth-reco association work.

## Motivation

CMS currently exposes truth information through several low-level collections, such as HepMC, GenParticles, SimTracks, SimVertices, TrackingParticles, SimClusters, CaloParticles, SimHits, and detector-specific RecHits. These collections are useful, but they encode different views of the event history and are often tied to detector-specific or production-specific conventions.

This package explores a different model: a single event-level truth graph that can be navigated using physics concepts, with optional detector-hit indices layered on top.

Typical questions this should make easier are:

* Do two reconstructed objects come from the same parent particle?
* Did a given resonance, such as a Z boson, exist in the event history?
* Do two reconstructed objects come from the same Z boson?
* Which parton initiated a reconstructed jet?
* Is an object associated with the hard interaction or with pileup?
* Which detector-level interactions contributed to a reconstructed object?
* Which SimHits were produced directly by a particle?
* Which SimHits were produced by the full subgraph starting from a particle?
* Which RecHits correspond to those SimHits through a DetId association?
* Should a reconstructed object be associated to a single truth particle, to a branch, or to an aggregated subgraph?

The intended user-facing API should allow reconstruction and validation code to operate on stable physics abstractions rather than directly depending on the storage details of `GenParticle`, `SimTrack`, `GenVertex`, `SimVertex`, `PCaloHit`, or detector-specific RecHit collections.

## Package layout

```text
PhysicsTools/TruthInfo/
  interface/
    TruthGraph.h
    Graph.h
    LogicalGraphHitIndex.h
    LogicalGraphHitIndexBuilder.h
  src/
    TruthGraph.cc
    Graph.cc
    LogicalGraphHitIndexBuilder.cc
    classes.h
    classes_def.xml
  plugins/
    TruthGraphProducer.cc
    TruthGraphDumper.cc
    TruthLogicalGraphProducer.cc
    TruthLogicalGraphDumper.cc
    TruthLogicalGraphHitIndexProducer.cc
    BuildFile.xml
  python/
    truthGraphProducer_cfi.py
    truthLogicalGraphDumper_cfi.py
  BuildFile.xml
````

The auxiliary RecHit lookup used by the hit index is produced separately in:

```text
SimCalorimetry/HGCalAssociatorProducers/
  interface/
    DetIdRecHitMap.h
  plugins/
    SimHitToRecHitMapProducer.cc
```

Despite living under `HGCalAssociatorProducers`, the current `SimHitToRecHitMapProducer` is not HGCal-only. It accepts both `HGCRecHitCollection` inputs and `reco::PFRecHitCollection` inputs.

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
* `SimToGen` is reserved.

The DOT dumper labels cross-domain edges explicitly, for example:

```text
GenToSim
SimToGen
```

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
std::vector<uint16_t> statusFlags;
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

Generator-to-simulation particle associations are built using the available `SimTrack` to generator information. The implementation keeps this association explicit in the raw graph instead of assuming that raw generator iteration indices can always be interpreted as stable GenParticle indices.

When enabled, cross-domain edges are also added:

* `GenParticle -> SimTrack`;
* `GenVertex -> SimVertex`, using the production vertex of the associated generator particle when available.

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
  uint16_t statusFlags = 0;

  uint64_t eventId = 0;
  int32_t genEvent = -1;

  math::XYZTLorentzVectorD momentum;
  std::vector<Checkpoint> checkpoints;

  bool hasGen() const;
  bool hasSim() const;
  bool valid() const;
};
```

The convention for the momentum is "best available":

1. for GEN+SIM particles, use the GEN four-momentum;
2. for SIM-only particles, use the `SimTrack` four-momentum;
3. otherwise keep the default value.

Useful methods include:

```cpp
bool hasGen() const;
bool hasSim() const;
int32_t pdgId() const;
int16_t status() const;
uint16_t statusFlags() const;
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

  bool hasGen() const;
  bool hasSim() const;
  bool valid() const;
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

Generator-level and simulation-level particles may be merged when a robust association exists.

Generator-level and simulation-level vertices can be merged by configuration when the producer has enough information to do so:

```python
mergeGenSimVertices = cms.bool(True)
```

This is useful for compact visualization and for downstream navigation. During debugging, it can be disabled to inspect generator and simulation vertex semantics separately.

### Intermediate GEN particle collapsing

The logical graph producer can collapse simple generator-only chains of the form:

```text
P -> V -> C
```

when:

* `P` is not final-state;
* `C` is the only daughter;
* `P` and `C` have the same PDG id.

This is controlled by:

```python
collapseIntermediateGenParticles = cms.bool(True)
```

This helps reduce visual clutter from intermediate generator copies while preserving the physically relevant branching structure.

### Selecting the interesting physics subgraph

The logical graph can optionally be restricted to a physics selection configured in the `postProcessing` PSet:

```python
postProcessing = cms.PSet(
    seedPdgIds = cms.vint32(23),
    seedParentDepth = cms.uint32(1),
    decayPdgIdGroups = cms.VPSet(
        cms.PSet(pdgIds = cms.vint32(13, -13)),
    ),
)
```

Selection by seed PDG id (`seedPdgIds`):

* the most upstream particle of each matching chain becomes a root of the selected graph (for a `Z0 -> Z0 -> Z0` chain, the earliest copy);
* the full downstream subgraph of each root is kept;
* `seedParentDepth` generations of ancestors are kept above each root as context only, without their other descendants;
* stable status-1 GEN particles outside the selection are always kept, attached to one artificial source vertex, unless explicitly ignored;
* kept particles whose production vertices all fall outside the selection are attached to the same artificial source vertex, which conceptually represents ISR or other uninteresting upstream activity;
* the special value `0` disables the selection and keeps the full graph (debugging escape hatch);
* an empty list applies no seed-based cut.

Selection by decay pattern (`decayPdgIdGroups`):

* each group is an unordered, charge-sensitive multiset of PDG ids (`[13, -13]` differs from `[13, 13]`); groups are OR-ed;
* matching is local to one decay vertex after following same-PDG radiating copy chains (`Z -> Z gamma`), with extra products such as FSR photons allowed; unrelated particles from different branches can never be combined, and `Z -> tau tau -> mu nu nu mu nu nu` does not match `[13, -13]`.

Combination semantics:

* only `seedPdgIds`: keep all decays of the selected roots;
* only `decayPdgIdGroups`: select vertices whose outgoing PDG ids contain a group, keep the matched particles, their downstream subgraphs, and the matched vertex as common production context;
* both: keep only roots whose effective decay products match a group (`Z -> mu mu` but not `Z -> e e`); if the event contains no particle with a seed PDG id at all, fall back to the direct vertex search (for generators that do not write the resonance explicitly);
* if a selection is configured but nothing matches, the output contains only the stable GEN particles attached to the artificial source vertex, and a warning is logged.

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
4. Merge particle nodes across GEN and SIM when a robust association exists.
5. Optionally merge GEN and SIM vertices.
6. Optionally collapse intermediate GEN-only particle copies.
7. Fill standalone `ParticleData` and `VertexData` payload.
8. Rebuild the logical bipartite graph in CSR-like adjacency vectors.
9. Validate the produced `truth::Graph`.

The produced graph is independent of the raw graph for ordinary navigation, but it keeps optional back-references to raw node ids for debugging:

```cpp
ParticleData::genNode
ParticleData::simNode
VertexData::genNode
VertexData::simNode
```

## Logical hit index: `truth::LogicalGraphHitIndex`

`truth::LogicalGraphHitIndex` is an auxiliary data product that associates logical graph particles to calorimeter SimHits and, when possible, to RecHits.

The key idea is that SimHits are directly associated to the particle that produced them, while subgraph hit collections are computed by aggregating over descendants.

For each logical particle, the hit index can answer two different questions:

1. Which SimHits were produced directly by this particle?
2. Which SimHits were produced by this particle and by all particles in the subgraph below it?

This is important because both views are useful:

* direct hits preserve local detector contributions from one particle;
* subgraph hits represent the full detector footprint of a shower, decay branch, or composite truth object.

### Hit payload

The hit index stores compact hit records:

```cpp
struct Hit {
  uint32_t detId;
  uint32_t recHitIndex;
  float energy;
};
```

The `detId` is the reco DetId used for matching to RecHits.

The `recHitIndex` is the index in the global RecHit ordering produced by `SimHitToRecHitMapProducer`. If no RecHit was found for the SimHit DetId, it is set to:

```cpp
truth::LogicalGraphHitIndex::Hit::invalidRecHitIndex
```

The `energy` is the accumulated SimHit energy for that logical particle and DetId.

### Direct and subgraph access

The hit index exposes spans of hits per logical particle, for example:

```cpp
auto directHits = hitIndex.directHits(particleId);
auto subgraphHits = hitIndex.subgraphHits(particleId);
```

The direct hits are attached only to the particle that directly produced them.

The subgraph hits are accumulated from the particle and all its descendants in the logical graph.

This makes merging two particles or two subgraphs straightforward: one can merge the corresponding hit spans by DetId or by RecHit index, depending on the intended matching metric.

## `TruthLogicalGraphHitIndexProducer`

`TruthLogicalGraphHitIndexProducer` builds `truth::LogicalGraphHitIndex`.

It consumes:

* the logical `truth::Graph`;
* the raw `TruthGraph`;
* selected `PCaloHit` collections;
* an optional DetId to RecHit index map produced by `SimHitToRecHitMapProducer`.

Typical configuration:

```python
process.truthLogicalGraphHitIndexProducer = cms.EDProducer(
    "TruthLogicalGraphHitIndexProducer",

    src = cms.InputTag("truthLogicalGraphProducer"),
    rawSrc = cms.InputTag("truthGraphProducer"),

    recHitMap = cms.InputTag("simHitToRecHitMapProducer"),

    simHitCollections = cms.VInputTag(
        cms.InputTag("g4SimHits", "HGCHitsEE", "SIM"),
        cms.InputTag("g4SimHits", "HGCHitsHEfront", "SIM"),
        cms.InputTag("g4SimHits", "HGCHitsHEback", "SIM"),
        cms.InputTag("g4SimHits", "EcalHitsEB", "SIM"),
        cms.InputTag("g4SimHits", "HcalHits", "SIM"),
    ),

    doHGCalRelabelling = cms.bool(False),
)
```

The producer performs the following steps:

1. Build a `SimTrack::trackId()` to logical particle id map.
2. Read the configured `PCaloHit` collections.
3. Convert SimHit DetIds to reco DetIds when requested.
4. Look up the corresponding RecHit index through the DetId map, if available.
5. Fill direct hits for the logical particle associated to the Geant track id.
6. Propagate and merge hit collections upward to build subgraph hit collections.

The hit index is intentionally separate from `truth::Graph`. This keeps the graph compact and allows detector-specific hit indices to evolve independently.

## `SimHitToRecHitMapProducer`

`SimHitToRecHitMapProducer` builds the DetId to RecHit index lookup consumed by `TruthLogicalGraphHitIndexProducer`.

The produced type is:

```cpp
hgcal::DetIdRecHitMap
```

with the current alias defined in:

```cpp
SimCalorimetry/HGCalAssociatorProducers/interface/DetIdRecHitMap.h
```

Conceptually it is:

```cpp
std::unordered_map<uint32_t, uint32_t>
```

mapping:

```text
reco DetId rawId -> global RecHit index
```

The global RecHit index is built by concatenating the configured RecHit collections in a deterministic order:

1. all configured `HGCRecHitCollection` inputs;
2. all configured `reco::PFRecHitCollection` inputs.

Typical configuration for RECO step3 output:

```python
process.simHitToRecHitMapProducer = cms.EDProducer(
    "SimHitToRecHitMapProducer",

    hgcalRecHits = cms.VInputTag(
        cms.InputTag("HGCalRecHit", "HGCEERecHits", "RECO"),
        cms.InputTag("HGCalRecHit", "HGCHEFRecHits", "RECO"),
        cms.InputTag("HGCalRecHit", "HGCHEBRecHits", "RECO"),
    ),

    pfRecHits = cms.VInputTag(
        cms.InputTag("particleFlowRecHitECAL", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHBHE", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHF", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHO", "Cleaned", "RECO"),
    ),
)
```

Do not include both `HGCalRecHit` and `particleFlowRecHitHGC` unless the intended indexing and double-counting policy is explicit. In the current workflow, HGCAL RecHits are taken from `HGCalRecHit`, while barrel and forward PF RecHits are taken from the cleaned `particleFlowRecHit*` collections.

The map needs a ROOT dictionary because it is an EDM product, even if it is not written to the output file.

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

### Access direct and subgraph hits

```cpp
auto const& hitIndex = event.get(hitIndexToken_);

for (uint32_t particleId = 0; particleId < hitIndex.nParticles(); ++particleId) {
  auto directHits = hitIndex.directHits(particleId);
  auto subgraphHits = hitIndex.subgraphHits(particleId);

  float directEnergy = 0.f;
  for (auto const& hit : directHits) {
    directEnergy += hit.energy;
  }

  float subgraphEnergy = 0.f;
  for (auto const& hit : subgraphHits) {
    subgraphEnergy += hit.energy;
  }
}
```

## Dumping and visualization

Two graph dumper modules are provided.

### Raw graph dumper

`TruthGraphDumper` writes a DOT representation of the raw graph.

It includes enriched labels using HepMC and simulation products when available.

Default output:

```text
truthgraph.dot
```

The dumper can be configured with:

```python
process.truthGraphDumper = cms.EDAnalyzer(
    "TruthGraphDumper",
    src = cms.InputTag("truthGraphProducer"),
    dotFile = cms.string("truthgraph.dot"),
    maxNodes = cms.uint32(5000),
    maxEdgesPerNode = cms.uint32(200),

    simTracks = cms.InputTag("g4SimHits"),
    simVertices = cms.InputTag("g4SimHits"),
    genEventHepMC = cms.InputTag("generatorSmeared"),
    genEventHepMC3 = cms.InputTag("generatorSmeared"),
)
```

### Logical graph dumper

`TruthLogicalGraphDumper` writes a DOT representation of the logical graph.

Default output:

```text
truthlogicalgraph.dot
```

The dumper can also use:

* the raw `TruthGraph`, to enrich labels with raw node information;
* the `LogicalGraphHitIndex`, to annotate particles with direct and subgraph SimHit summaries;
* the RecHit collections, to compute RecHit energy summaries from `recHitIndex`.

Example:

```python
process.truthLogicalGraphDumper = cms.EDAnalyzer(
    "TruthLogicalGraphDumper",
    src = cms.InputTag("truthLogicalGraphProducer"),
    rawSrc = cms.InputTag("truthGraphProducer"),
    hitIndex = cms.InputTag("truthLogicalGraphHitIndexProducer"),

    hgcalRecHits = cms.VInputTag(
        cms.InputTag("HGCalRecHit", "HGCEERecHits", "RECO"),
        cms.InputTag("HGCalRecHit", "HGCHEFRecHits", "RECO"),
        cms.InputTag("HGCalRecHit", "HGCHEBRecHits", "RECO"),
    ),

    pfRecHits = cms.VInputTag(
        cms.InputTag("particleFlowRecHitECAL", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHBHE", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHF", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHO", "Cleaned", "RECO"),
    ),

    dotFile = cms.string("truthlogicalgraph.dot"),

    maxParticles = cms.uint32(5000),
    maxVertices = cms.uint32(5000),
    maxEdgesPerNode = cms.uint32(200),

    hideLargeSimSourceVertices = cms.bool(True),
    largeSimSourceVertexMinOutgoing = cms.uint32(50),
)
```

Particle labels include summaries such as:

```text
direct simHits: N  simE=...
direct recHits: N  missing=...  recoE=...
subgraph simHits: N  simE=...
subgraph recHits: N  missing=...  recoE=...
```

DOT files can be converted with Graphviz, for example:

```bash
dot -Tsvg truthlogicalgraph_run1_lumi1_event1.dot -o truthlogicalgraph_run1_lumi1_event1.svg
```

To convert all DOT files in a directory:

```bash
for f in *.dot; do dot -Tsvg "$f" -o "${f%.dot}.svg"; done
```

To inspect whether hit information is present in the DOT output:

```bash
grep -n "simHits\|recHits\|simE\|recoE" truthlogicalgraph_run1_lumi1_event1.dot | head -80
```

## Example configuration on step3.root

A typical standalone test configuration running on a `step3.root` file is:

```python
import FWCore.ParameterSet.Config as cms

process = cms.Process("TRUTHGRAPH")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.Geometry.GeometryExtendedRun4D120Reco_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring("file:step3.root")
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.truthGraphProducer = cms.EDProducer(
    "TruthGraphProducer",
    genEventHepMC3 = cms.InputTag("generatorSmeared"),
    genEventHepMC = cms.InputTag("generatorSmeared"),
    simTracks = cms.InputTag("g4SimHits"),
    simVertices = cms.InputTag("g4SimHits"),
    addGenToSimEdges = cms.bool(True),
)

process.truthGraphDumper = cms.EDAnalyzer(
    "TruthGraphDumper",
    src = cms.InputTag("truthGraphProducer"),
    dotFile = cms.string("truthgraph.dot"),
    maxNodes = cms.uint32(20000),
    maxEdgesPerNode = cms.uint32(50),
    simTracks = cms.InputTag("g4SimHits"),
    simVertices = cms.InputTag("g4SimHits"),
    genEventHepMC = cms.InputTag("generatorSmeared"),
    genEventHepMC3 = cms.InputTag("generatorSmeared"),
)

process.truthLogicalGraphProducer = cms.EDProducer(
    "TruthLogicalGraphProducer",
    src = cms.InputTag("truthGraphProducer"),
    simTracks = cms.InputTag("g4SimHits"),
    simVertices = cms.InputTag("g4SimHits"),
    genEventHepMC3 = cms.InputTag("generatorSmeared"),
    genEventHepMC = cms.InputTag("generatorSmeared"),

    motherPdgId = cms.int32(0),
    mergeGenSimVertices = cms.bool(True),
    collapseIntermediateGenParticles = cms.bool(True),
)

process.simHitToRecHitMapProducer = cms.EDProducer(
    "SimHitToRecHitMapProducer",

    hgcalRecHits = cms.VInputTag(
        cms.InputTag("HGCalRecHit", "HGCEERecHits", "RECO"),
        cms.InputTag("HGCalRecHit", "HGCHEFRecHits", "RECO"),
        cms.InputTag("HGCalRecHit", "HGCHEBRecHits", "RECO"),
    ),

    pfRecHits = cms.VInputTag(
        cms.InputTag("particleFlowRecHitECAL", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHBHE", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHF", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHO", "Cleaned", "RECO"),
    ),
)

process.truthLogicalGraphHitIndexProducer = cms.EDProducer(
    "TruthLogicalGraphHitIndexProducer",

    src = cms.InputTag("truthLogicalGraphProducer"),
    rawSrc = cms.InputTag("truthGraphProducer"),

    recHitMap = cms.InputTag("simHitToRecHitMapProducer"),

    simHitCollections = cms.VInputTag(
        cms.InputTag("g4SimHits", "HGCHitsEE", "SIM"),
        cms.InputTag("g4SimHits", "HGCHitsHEfront", "SIM"),
        cms.InputTag("g4SimHits", "HGCHitsHEback", "SIM"),
        cms.InputTag("g4SimHits", "EcalHitsEB", "SIM"),
        cms.InputTag("g4SimHits", "HcalHits", "SIM"),
    ),

    doHGCalRelabelling = cms.bool(False),
)

process.truthLogicalGraphDumper = cms.EDAnalyzer(
    "TruthLogicalGraphDumper",
    src = cms.InputTag("truthLogicalGraphProducer"),
    rawSrc = cms.InputTag("truthGraphProducer"),
    hitIndex = cms.InputTag("truthLogicalGraphHitIndexProducer"),

    hgcalRecHits = cms.VInputTag(
        cms.InputTag("HGCalRecHit", "HGCEERecHits", "RECO"),
        cms.InputTag("HGCalRecHit", "HGCHEFRecHits", "RECO"),
        cms.InputTag("HGCalRecHit", "HGCHEBRecHits", "RECO"),
    ),

    pfRecHits = cms.VInputTag(
        cms.InputTag("particleFlowRecHitECAL", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHBHE", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHF", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHO", "Cleaned", "RECO"),
    ),

    dotFile = cms.string("truthlogicalgraph.dot"),

    maxParticles = cms.uint32(20000),
    maxVertices = cms.uint32(20000),
    maxEdgesPerNode = cms.uint32(300),
)

process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.cerr.default = cms.untracked.PSet(
    limit = cms.untracked.int32(0)
)
process.MessageLogger.cerr.TruthGraphProducer = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)
process.MessageLogger.cerr.TruthLogicalGraphProducer = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)
process.MessageLogger.cerr.TruthLogicalGraphHitIndexProducer = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)
process.MessageLogger.cerr.SimHitToRecHitMapProducer = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)

process.truthGraph_step = cms.Path(
    process.truthGraphProducer +
    process.truthGraphDumper +
    process.truthLogicalGraphProducer +
    process.simHitToRecHitMapProducer +
    process.truthLogicalGraphHitIndexProducer +
    process.truthLogicalGraphDumper
)
```

## Event content checks

Useful commands to inspect available products are:

```bash
edmDumpEventContent step3.root | grep -E 'PFRecHit|particleFlowRecHit'
```

and:

```bash
edmDumpEventContent step3.root | grep -E 'HGCRecHit|HGCalRecHit|HGCEERecHits|HGCHEFRecHits|HGCHEBRecHits'
```

For a typical Phase-2 RECO file, the useful collections are:

```text
vector<HGCRecHit>       "HGCalRecHit"            "HGCEERecHits"     "RECO"
vector<HGCRecHit>       "HGCalRecHit"            "HGCHEFRecHits"    "RECO"
vector<HGCRecHit>       "HGCalRecHit"            "HGCHEBRecHits"    "RECO"

vector<reco::PFRecHit>  "particleFlowRecHitECAL" "Cleaned"          "RECO"
vector<reco::PFRecHit>  "particleFlowRecHitHBHE" "Cleaned"          "RECO"
vector<reco::PFRecHit>  "particleFlowRecHitHF"   "Cleaned"          "RECO"
vector<reco::PFRecHit>  "particleFlowRecHitHO"   "Cleaned"          "RECO"
```

The relevant SimHit collections include:

```text
vector<PCaloHit>        "g4SimHits"              "HGCHitsEE"        "SIM"
vector<PCaloHit>        "g4SimHits"              "HGCHitsHEfront"   "SIM"
vector<PCaloHit>        "g4SimHits"              "HGCHitsHEback"    "SIM"
vector<PCaloHit>        "g4SimHits"              "EcalHitsEB"       "SIM"
vector<PCaloHit>        "g4SimHits"              "HcalHits"         "SIM"
vector<SimTrack>        "g4SimHits"              ""                 "SIM"
vector<SimVertex>       "g4SimHits"              ""                 "SIM"
```

## Intended matching strategy

The long-term matching model is to build detector-level associators from hits to `truth::Particle`.

In this model:

* the graph provides the particle and vertex structure;
* detector-level truth association is performed through hits;
* reconstructed objects can be associated either to a single `truth::Particle` or to a larger truth branch;
* truth information can be aggregated over a coherent branch when a reconstructed object corresponds to a composite truth structure.

Different reconstruction domains need different matching metrics:

* tracking association is usually based on shared hits, not on hit energy;
* calorimeter clustering association can use energy fractions or energy-weighted metrics;
* timing objects may need time-aware matching;
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
* define stable references for composite truth structures;
* merge two selected truth structures and their hit content.

In this picture, `truth::Branch` avoids forcing an early collapse of the truth history into fixed reference objects.

## Hits attached to particles

The current prototype already implements the first version of hit attachment through `truth::LogicalGraphHitIndex`.

This keeps hit information outside the main logical graph data product, while allowing the graph dumper and future associators to query:

* direct SimHits from a particle;
* subgraph SimHits from a particle and all descendants;
* matched RecHit indices;
* SimHit energy sums;
* RecHit energy sums.

A possible long-term direction is to generalize this further so that detector-specific truth objects such as `TrackingParticle`, `SimCluster`, and `CaloParticle` can be expressed as views or derived abstractions on top of the same graph and hit-index infrastructure.

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
* optionally merge generator and simulation vertices;
* optionally collapse intermediate GEN-only particle copies;
* navigate particle and vertex relations;
* compute parents, children, ancestors, descendants, roots, and leaves;
* find ancestors with a given PDG id;
* find a common ancestor between two particles;
* store boundary-crossing checkpoints;
* build a direct and subgraph calorimeter SimHit index per logical particle;
* map SimHits to RecHit indices through DetId when a RecHit map is available;
* dump raw and logical graphs to DOT for debugging;
* annotate logical graph DOT nodes with SimHit and RecHit energy summaries.

## Known limitations

The current implementation is a prototype and several aspects are intentionally conservative.

Known limitations include:

* GEN-SIM association still needs broader validation;
* the semantics of vertex merging require further study;
* checkpoint information is currently limited to boundary-crossing information from `SimTrack`;
* the hit index currently targets calorimeter `PCaloHit` inputs;
* RecHit association is currently DetId-based and does not encode more detailed detector response information;
* duplicate DetId handling depends on the configured RecHit input ordering and map policy;
* `truth::Branch` is part of the target design but not yet implemented;
* the logical API is still evolving;
* the raw graph construction needs further validation on realistic events and pileup scenarios.

## Next steps

Planned or natural next steps are:

1. Continue debugging the raw truth graph construction.
2. Validate HepMC2 and HepMC3 behaviour on representative workflows.
3. Refine GEN-SIM particle association.
4. Study the vertex merging policy in realistic events.
5. Extend the checkpoint model beyond boundary crossings.
6. Investigate direct graph construction during Geant4 simulation.
7. Extend hit indexing beyond calorimeter `PCaloHit` where appropriate.
8. Prototype detector-specific truth-reco matching metrics on top of the hit index.
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
hitIndex.directHits(particle.id());
hitIndex.subgraphHits(particle.id());
```

rather than reimplementing event-history navigation and hit aggregation separately for each reconstruction or validation use case.


