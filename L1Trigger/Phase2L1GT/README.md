# Menu configuration manual
# Small Addition for testing purposes
**NB! A general overview of the P2GT emulator menus is given in the `L1Trigger/Configuration/python/Phase2GTMenus/README.md`**

## Conditions

The basic building blocks of a Phase-2 Global Trigger menu are conditions. In order to start writing a menu one should first pull the standard definitions for the conditions into the configuration. These standard definitions contain the scale parameters (c.f. [l1tGTScales.py](python/l1tGTScales.py)) as well as the values for the $\cos$ and $\cosh$ LUT (computed in [l1tGTSingleInOutLUT.py](python/l1tGTSingleInOutLUT.py)).

```python
from L1Trigger.Phase2L1GT.l1tGTSingleObjectCond_cfi import l1tGTSingleObjectCond
from L1Trigger.Phase2L1GT.l1tGTDoubleObjectCond_cfi import l1tGTDoubleObjectCond
from L1Trigger.Phase2L1GT.l1tGTTripleObjectCond_cfi import l1tGTTripleObjectCond
from L1Trigger.Phase2L1GT.l1tGTQuadObjectCond_cfi import l1tGTQuadObjectCond
```
One can utilize the standard definitions by invoking the `clone` function and specifying a cut configuration as well as the collection(s) the condition should act on. This is done by setting the corresponding input tag(s) to use, `tag = cms.InputTag("l1tGTProducer", "XXX")`, with `XXX` as the collection name. Available collections are:

| GCT | GMT |  GTT | Correlator Layer 2 |
|:-----|:----------|:-------------|:--------|
| ~~`GCTNonIsoEg`~~ |  `GMTSaPromptMuons` | `GTTPromptJets` | `CL2JetsSC4` |
| ~~`GCTIsoEg`~~ | `GMTSaDisplacedMuons` | `GTTDisplacedJets` | `CL2JetsSC8` |
| ~~`GCTJets`~~ |  `GMTTkMuons` | ~~`GTTPhiCandidates`~~ | `CL2Taus` |
| ~~`GCTTaus`~~ | ~~`GMTTopo`~~ | ~~`GTTRhoCandidates`~~ | `CL2Electrons` |
| ~~`GCTHtSum`~~ |  | ~~`GTTBsCandidates`~~ | `CL2Photons` |
| ~~`GCTEtSum`~~ |  | ~~`GTTHadronicTaus`~~ | `CL2HtSum` |
| | | `GTTPrimaryVert` | `CL2EtSum` |
| | | ~~`GTTPromptHtSum`~~ | |
| | | ~~`GTTDisplacedHtSum`~~ | |
| | | ~~`GTTEtSum`~~ | |

~~`XXX`~~: Not yet available from upstream emulator.

A condition can have any number of cuts. Cuts omitted  from the configuration are regarded as disabled. They can either be applied to single objects of a collection or to topological correlations within one or between multiple collections. In general the configuration follows an ambiguity scheme, as long as there are no ambiguities the configuration should be specified in the top level `PSet`. If there are ambiguities the configuration requires either a `collectionX` sub `PSet` for single object cuts or a `correlXY`/`correlXYZ` sub `PSet` for correlations. For illustration the following shows some examples:

```python
process.SingleTkMuon22 = l1tGTSingleObjectCond.clone(
    # No ambiguities, thus everything in the top level PSet
    tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
    maxAbsEta = cms.double(2.4),
    regionsAbsEtaLowerBounds = cms.vdouble(0, 0.83, 1.24),
    regionsMinPt = cms.vdouble(20, 20, 20)
)
```

```python
process.DoubleTkEle2512 = l1tGTDoubleObjectCond.clone(
    # 2 single cuts, thus cuts are ambiguous and require collectionX PSets.
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        minPt = cms.double(20),
        maxAbsEta = cms.double(2.4),
        regionsAbsEtaLowerBounds = cms.vdouble(0, 1.479),
        regionsQualityFlags = cms.vuint32(0b0010, 0b0000)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "CL2Electrons"),
        minPt = cms.double(9),
        maxAbsEta = cms.double(2.4),
        regionsAbsEtaLowerBounds = cms.vdouble(0, 1.479),
        regionsQualityFlags = cms.vuint32(0b0010, 0b0000)
    ),
    # Correlation can only be between 1 and 2 -> no ambiguities 
    maxDz = cms.double(1),
)
```

```python
process.TripleTkMuon533 = l1tGTTripleObjectCond.clone(
    # 3 single cuts, thus cuts are ambiguous and require collectionX PSets.
    collection1 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minPt = cms.double(5),
        maxAbsEta = cms.double(2.4),
        qualityFlags = cms.uint32(0b0001)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minPt = cms.double(3),
        maxAbsEta = cms.double(2.4),
        qualityFlags = cms.uint32(0b0001)
    ),
    collection3 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minPt = cms.double(3),
        maxAbsEta = cms.double(2.4),
        qualityFlags = cms.uint32(0b0001)
    ),
    # Correlations are ambiguous (can be {1,2}, {1,3}, or {2,3}), correlXY PSets are thus required.
    correl12 = cms.PSet(
        maxDz = cms.double(1)
    ),
)
```

### Object presets to synchronise with the MenuTools

The L1 DPG Phase-2 Menu team is responsible for implementing, maintaining and validating the L1 menu.
See the twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PhaseIIL1TriggerMenuTools

In `L1Trigger/Phase2L1GT/python/l1tGTObject_constants.py` there are several functions to extract object cut properties defined by the menu team, which allow to simplify and synchronise the seed definitions between the P2GT emulator in CMSSW and the [MenuTools](https://github.com/cms-l1-dpg/Phase2-L1MenuTools).

There are these getter functions available to simplify the seed definitions:
* `get_object_etalowbounds(obj)` to get the min abs(eta) requirements to be set for `regionsAbsEtaLowerBounds`
* `get_object_thrs(thr, obj, id)` to set the thresholds using the online-offline scalings (per regions if available, otherwise a single value e.g. for sums). The `thr` argument is the offline threshold.
* `get_object_ids(obj, id)` to set the ID values (per regions if available) for 
* `get_object_isos(obj, id)`  to set the ID values (per regions if available)

The arguments are:
- `obj`: the trigger object name as from the P2GT producer, e.g. `GMTTkMuons`, 
- `id`: the ID label e.g. `Loose` or `NoIso`.

The definitions of the object requirements are hardcoded in these files:
* IDs: `L1Trigger/Phase2L1GT/python/l1tGTObject_ids.py`
* Scalings: `L1Trigger/Phase2L1GT/python/l1tGTObject_scalings.py`

An example of translating the hard-coded values for `GMTTkMuon` is below:
```python
collection1 = cms.PSet(
    tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
    regionsAbsEtaLowerBounds=cms.vdouble(0,0.83,1.24),
    regionsMinPt=cms.vdouble(13,13,13)
    qualityFlags = cms.uint32(0b0001)
    ...
)
```

Becomes:
```python
collection1 = cms.PSet(
    tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
    regionsAbsEtaLowerBounds = get_object_etalowbounds("GMTTkMuons"),
    regionsMinPt = get_object_thrs(15, "GMTTkMuons","VLoose"),
    qualityFlags = get_object_ids("GMTTkMuons","VLoose"),
    ...
)
```

#### Simplification for common baseline objects

As there are only few baseline objects which are are used in many seeds, it might be simpler to define some baseline objects that could be re-used in many seeds, modifying/extending only with additional cuts. 

E.g. in the b-physics seeds identical `Loose` `tkMuons` are used with only the pt thresholds varying.
Thus a baseline tkMuon can be defined as:
```python
 l1tGTtkMuon = cms.PSet(
    tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
    minEta = cms.double(-2.4),
    maxEta = cms.double(2.4),
    regionsAbsEtaLowerBounds = get_object_etalowbounds("GMTTkMuons"),
)
l1tGTtkMuonVLoose = l1tGTtkMuon.clone(
    qualityFlags = get_object_ids("GMTTkMuons","VLoose"),
)
```
And then this can be used in seed definitions by only changing / adding what is needed, e.g.:
```python
FakeDiMuSeed = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTtkMuon.clone(
        minPt = cms.double(5), # overwrites minPt = 0 from template
    ),
    collection2 = l1tGTtkMuonVLoose.clone(
        minEta = cms.double(-1.5), # overwrites minEta = -2.4 
        maxEta = cms.double(1.5), # overwrites maxEta = 2.4 
    ),
)
```
**NB!** Also new cuts can be added, however caution is needed to not mix per-region cuts such as `regionsMinPt` and the inclusive version `minPt`.

For single object conditions using pre-defined objects requires to first clone the condition and then extend it with the objects PSet as shown below:
```python
SingleTkMuon22 = l1tGTSingleObjectCond.clone(
    l1tGTtkMuonVLoose.clone(),
)    
```

### Single cuts

Possible cuts on single quantities are:

| Name | Expression |     Datatype |  Hardware conversion |
|:-----|:----------:|:-------------:|:--------:|
| `minPt`    |  $p_T > X$ or $\| \sum \vec p_T \| > X$ | `cms.double` | `floor(X / pT_lsb)` |
| `maxPt`    |  $p_T < X$ or $\| \sum \vec p_T \| < X$ | `cms.double` | `ceil(X / pT_lsb)` |
| `minEta`   |  $\eta > X$ | `cms.double` | `floor(X / eta_lsb)` |
| `maxEta`   | $\eta < X$ | `cms.double` | `ceil(X / eta_lsb)` |
| `minPhi`   | $\phi > X$ | `cms.double` | `floor(X / phi_lsb)` |
| `maxPhi`   | $\phi < X$ | `cms.double` | `ceil(X / phi_lsb)` |
| `minZ0`   | $z_0 > X$ | `cms.double` | `floor(X / z0_lsb)` |
| `maxZ0`   | $z_0 < X$ | `cms.double` | `ceil(X / z0_lsb)` |
| `minScalarSumPt`   | $\sum p_T > X$ | `cms.double` | `floor(X / scalarSumPT_lsb)` |
| `maxScalarSumPt`   | $\sum p_T < X$ | `cms.double` | `ceil(X / scalarSumPT_lsb)` |
| `minQualityScore`   | $\mathrm{qualityScore} > X$ | `cms.uint32` | `X` |
| `maxQualityScore`   | $\mathrm{qualityScore} < X$ | `cms.uint32` | `X` |
| `qualityFlags`   |  $\mathrm{qualityFlags} \wedge X = X$ | `cms.uint32` | `X` |
| `minAbsEta`   | $\| \eta \| > X $ | `cms.double` | `floor(X / eta_lsb)` |
| `maxAbsEta`   | $\| \eta \| < X $ | `cms.double` | `ceil(X / eta_lsb)` |
| `minIsolationPt`   | $\mathrm{isolationPT} > X$ | `cms.double` | `floor(X / isolationPT_lsb)` |
| `maxIsolationPt`   | $\mathrm{isolationPT} < X$ | `cms.double` | `ceil(X / isolationPT_lsb)` |
| `minRelIsolationPt` | $\mathrm{isolationPT} > X \cdot p_T$ | `cms.double` | `floor(X * pT_lsb * 2**18 / isolationPT)` |
| `maxRelIsolationPt` |  $\mathrm{isolationPT} < X \cdot p_T$ | `cms.double` | `ceil(X * pT_lsb * 2**18 / isolationPT)` |
| `minPrimVertDz`* | $\| z_0 - Z_{0,i} \| > X $ | `cms.double` | `floor(X / z0_lsb)` |
| `maxPrimVertDz`* | $\| z_0 - Z_{0,i} \| < X $ | `cms.double` | `ceil(X / z0_lsb)` |
| `minPtMultiplicityCut`** | $\sum \left( p_T > X\right) \geq N$ | `cms.double` | `floor(X / pT_lsb)` |

\* : To select a $Z_0$ index $i$ from the `GTTPrimaryVert` collection for the comparison use `primVertex = cms.uint32(i)`. This parameter is mandatory when using a `maxPrimVertDz` cut.

\** : Requires additional parameter $N$ with `minPtMultiplicityN = cms.uint32(N)`.

### $\eta$-regional cuts

Certain cuts can also be specified $\eta$-region dependent, to allow different thresholds in different regions. In order to use this feature, one has to first provide the lower bounds for the regions via `regionsAbsEtaLowerBounds`. This parameter takes an `cms.vdouble`, whose length determines the number of $\eta$-regions. A region then ranges from the specified lower bound (inclusive) up to the next region's lower bound (exclusive). The last region's upper bound is always the maximum allowed $|\eta| = 2\pi$. One can use additional global $\eta$ or $|\eta|$ cuts to exclude large $|\eta|$ values, effectively overriding the last region's upper bound. The following cuts can be specified per each $\eta$-region:

| Name | Expression |     Datatype |  Hardware conversion |
|:-----|:----------:|:-------------:|:--------:|
| `regionsMinPt`    |  $p_T > X$ or $\| \sum \vec p_T \| > X$ | `cms.vdouble` | `floor(X / pT_lsb)` |
| `regionsQualityFlags`   |  $\mathrm{qualityFlags} \wedge X = X$ | `cms.vuint32` | `X` |
| `regionsMaxRelIsolationPt`   |  $\mathrm{isolationPT} < X \cdot p_T$ | `cms.vdouble` | `ceil(X * pT_lsb * 2**18 / isolationPT)` |

Note: The vector parameters for the $\eta$-region cuts must have the same length as the number of $\eta$-regions initially set via `regionsAbsEtaLowerBounds`.

### Correlational cuts

Cuts can also be performed on topological correlations. The following 2-body correlational cuts are available:

| Name | Expression | Datatype | Hardware conversion |
|:-----|:----------:|:-------------:|:--------:|
| `minDEta` | $\|\eta_1 - \eta_2\| > X$ | `cms.double` | `floor(X / eta_lsb)` |
| `maxDEta` | $\|\eta_1 - \eta_2\| < X$ | `cms.double` | `ceil(X / eta_lsb)` |
| `minDPhi` | $\Delta \phi > X$ | `cms.double` | `floor(X / phi_lsb)` |
| `maxDPhi` | $\Delta \phi < X$ | `cms.double` | `ceil(X / phi_lsb)` |
| `minDz` | $\|z_{0,1} - z_{0,2}\| > X$ | `cms.double` | `floor(X / z0_lsb)` |
| `maxDz` | $\|z_{0,1} - z_{0,2}\| < X$ | `cms.double` | `ceil(X / z0_lsb)` |
| `minDR` | $\Delta \phi ^2 + \Delta \eta^2 > X^2$ | `cms.double` | `floor(X**2 / eta_lsb**2)` |
| `maxDR` | $\Delta \phi ^2 + \Delta \eta^2 < X^2$ | `cms.double` | `ceil(X**2 / eta_lsb**2)` |
| `minInvMass` | $p_{T,1} \, p_{T,2} \left[ \cosh(\Delta \eta) - \cos(\Delta \phi) \right] > X^2/2$ | `cms.double` | `floor(X**2 * LUT_Scale / (2 * pT_lsb**2))` |
| `maxInvMass` |  $p_{T,1} \, p_{T,2} \left[ \cosh(\Delta \eta) - \cos(\Delta \phi) \right] < X^2/2$ | `cms.double` | `ceil(X**2 * LUT_Scale / (2 * pT_lsb**2))` |
| `minTransMass` | $p_{T,1} \, p_{T,2} \left[1 - \cos(\Delta \phi) \right] > X^2/2$ | `cms.double` |  `floor(X**2 * LUT_Scale / (2 * pT_lsb**2))` |
| `maxTransMass` | $p_{T,1} \, p_{T,2} \left[1 - \cos(\Delta \phi) \right] < X^2/2$ | `cms.double` |  `ceil(X**2 * LUT_Scale / (2 * pT_lsb**2))` |
| `minCombPt` | $p_{T,1}^2 + p_{T,2}^2 + 2 p_{T,1} \, p_{T,2}  \cos(\Delta \phi) > X^2$ | `cms.double` | `floor(X**2 * LUT_Scale / pT_lsb**2)` |
| `maxCombPt` |  $p_{T,1}^2 + p_{T,2}^2 + 2 p_{T,1} \, p_{T,2}  \cos(\Delta \phi) < X^2$ | `cms.double` | `ceil(X**2 * LUT_Scale / pT_lsb**2)` |
| `minInvMassOverDR` | $m^2/2 > X^2 \cdot \Delta R^2/2$ | `cms.double` | `floor(X**2 * LUT_Scale * 2**19 * eta_lsb**2 / (2 * pT_lsb**2))` | 
| `maxInvMassOverDR` | $m^2/2 < X^2 \cdot \Delta R^2/2$ | `cms.double` | `ceil(X**2 * LUT_Scale * 2**19 * eta_lsb**2 / (2 * pT_lsb**2))` | 
| `os` | $q_1 \neq q_2$ | `cms.bool` | |
| `ss` | $q_1 = q_2$  | `cms.bool` | |

Note: $\Delta \eta = |\eta_1 - \eta_2|$, $\Delta \phi$ is the smallest angle between two legs, also taking into account that $\phi$ wraps around i.e. $\phi = \pi = - \pi$.

The following 3-body correlational cuts are available:

| Name | Expression | Datatype | Hardware conversion |
|:-----|:----------:|:-------------:|:--------:|
| `minInvMass` | $\frac{m_{1,2}^2}{2} +  \frac{m_{1,3}^2}{2} +  \frac{m_{2,3}^2}{2} > \frac{X^2}{2}$ | `cms.double` | `floor(X**2 * LUT_Scale / (2 * pT_lsb**2))` |
| `maxInvMass` | $\frac{m_{1,2}^2}{2} +  \frac{m_{1,3}^2}{2} +  \frac{m_{2,3}^2}{2} < \frac{X^2}{2}$ | `cms.double` | `ceil(X**2 * LUT_Scale / (2 * pT_lsb**2))` |
| `minTransMass` | $\frac{m_{T,1,2}^2}{2} +  \frac{m_{T,1,3}^2}{2} +  \frac{m_{T,2,3}^2}{2} > \frac{X^2}{2}$ | `cms.double` |  `floor(X**2 * LUT_Scale / (2 * pT_lsb**2))` |
| `maxTransMass` | $\frac{m_{T,1,2}^2}{2} +  \frac{m_{T,1,3}^2}{2} +  \frac{m_{T,2,3}^2}{2} < \frac{X^2}{2}$ | `cms.double` |  `ceil(X**2 * LUT_Scale / (2 * pT_lsb**2))` |

## Algorithms

Conditions are combined to algorithms via the [`L1GTAlgoBlockProducer`](plugins/L1GTAlgoBlockProducer.cc). To configure this behavior, a `cms.PSet` algorithm configuration should be added to the `algorithms` `cms.VPset`, included via:

```python
from L1Trigger.Phase2L1GT.l1tGTAlgoBlockProducer_cff import algorithms
```
A minimal configuration just includes an `expression` of `cms.Path`s. The available configuration parameters are:

| Name | Datatype | Description |
|:-----|:----------:|:--------------|
| `name` | `cms.string` | A unique algorithm identifier (default: `expression`) |
| `expression` | `cms.string` | `cms.Path` expression (required) |
| `prescale` | `cms.double` | Prescale value: 0 or in [1, $2^{24} - 1$) (default: 1) |
| `prescalePreview` | `cms.double` | Prescale preview value: 0 or in [1, $2^{24} - 1$) (default: 1) |
| `bunchMask` | `cms.vuint32` | Vector of bunch crossing numbers to mask (default: empty) |
| `triggerTypes` | `cms.vint32` | Vector of trigger type numbers assigned to this algorithm (default: 1) |
| `veto` | `cms.bool` | Indicates whether the algorithm is a veto (default: false) |

Utilizing the examples from the [Conditions section](#conditions) one could define an algorithm as follows:

```python
from L1Trigger.Phase2L1GT.l1tGTAlgoBlockProducer_cff import algorithms

process.pSingleTkMuon22 = cms.Path(process.SingleTkMuon22)
process.pDoubleTkEle25_12 = cms.Path(process.DoubleTkEle2512)

algorithms.append(cms.PSet(expression = cms.string("pSingleTkMuon22 or pDoubleTkEle25_12")))
```

## Firmware pattern writers

There are 3 types of Global Trigger pattern writers currently implemented.

* `L1GTAlgoBoardWriter`: Used to write out the algorithm bits into 2 channels. With config

| Name | Datatype | Description |
|:-----|:----------:|:--------------|
| `filename` | `cms.string` | The filename prefix to use for pattern files (required) |
| `fileExtension` | `cms.string` | `txt`, `txt.gz` or `txt.xz` (default: `txt`) |
| `algoBlocksTag` | `cms.InputTag` | AlgoBlock producer input tag to use (required) |
| `maxFrames` | `cms.unit32` | Maximum number of frames (default: 1024) |
| `maxEvents` | `cms.unit32` | Maximum number of events (default: events that fit into `maxFrames`) |
| `channels` | `cms.vuint32` | Vector of 2 channel numbers for output (required) |
| `algoBitMask` | `cms.vuint64` | Vector of 9 64 bit masks (default: all set to 1) |
| `patternFormat` | `cms.string` | `APx`, `EMPv1`, `EMPv2` or `X2O` (default: `EMPv2`) |

* `L1GTFinOrBoardWriter`: Used to write out Final OR bits (beforeBxMaskAndPrescale, beforePrescale and final) each on a different channel for the low bits (0 - 575), mid bits (576 - 1151) and high bits (1152 - 1727). 9 channels in total + one channel for the passing Final OR trigger types. Config:

| Name | Datatype | Description |
|:-----|:----------:|:--------------|
| `filename` | `cms.string` | The filename prefix to use for pattern files (required) |
| `fileExtension` | `cms.string` | `txt`, `txt.gz` or `txt.xz` (default: `txt`) |
| `algoBlocksTag` | `cms.InputTag` | AlgoBlock producer input tag to use (required) |
| `maxFrames` | `cms.unit32` | Maximum number of frames (default: 1024) |
| `maxEvents` | `cms.unit32` | Maximum number of events (default: events that fit into `maxFrames`) |
| `channelsLow` | `cms.vuint32` | Vector of 3 channel numbers for low bits (0 - 575) (required) |
| `channelsMid` | `cms.vuint32` | Vector of 3 channel numbers for mid bits (576 - 1151) (required) |
| `channelsHigh` | `cms.vuint32` | Vector of 3 channel numbers for high bits (1152 - 1727) (required) |
| `channelFinOr` | `cms.uint32` | Channel for FinalOr trigger types (required) |
| `patternFormat` | `cms.string` | `APx`, `EMPv1`, `EMPv2` or `X2O` (default: `EMPv2`) |

* `L1GTObjectBoardWriter`: Used to write input and output object patterns using the upstream provided pack functions.

| Name | Datatype | Description |
|:-----|:----------:|:--------------|
| `filename` | `cms.string` | The filename prefix to use for pattern files (required) |
| `fileExtension` | `cms.string` | `txt`, `txt.gz` or `txt.xz` (default: `txt`) |
| `maxFrames` | `cms.unit32` | Maximum number of frames (default: 1024) |
| `maxEvents` | `cms.unit32` | Maximum number of events (default: events that fit into `maxFrames`) |
| `patternFormat` | `cms.string` | `APx`, `EMPv1`, `EMPv2` or `X2O` (default: `EMPv2`) |
| `bufferFileType`| `cms.string` | Either `input` or `output` (required) |
| `InputChannels.GCT_1` | `cms.vuint32` | Channels for GCT link 1 (required if `bufferFileType` = `input`) |
| `InputChannels.GMT_1` | `cms.vuint32` | Channels for GMT link 1 (required if `bufferFileType` = `input`) |
| `InputChannels.GTT_1` | `cms.vuint32` | Channels for GTT link 1 (required if `bufferFileType` = `input`) |
| `InputChannels.GTT_2` | `cms.vuint32` | Channels for GTT link 2 (required if `bufferFileType` = `input`) |
| `InputChannels.GTT_3` | `cms.vuint32` | Channels for GTT link 3 (required if `bufferFileType` = `input`) |
| `InputChannels.GTT_4` | `cms.vuint32` | Channels for GTT link 4 (required if `bufferFileType` = `input`) |
| `InputChannels.CL2_1` | `cms.vuint32` | Channels for CL2 link 1 (required if `bufferFileType` = `input`) |
| `InputChannels.CL2_2` | `cms.vuint32` | Channels for CL2 link 2 (required if `bufferFileType` = `input`) |
| `InputChannels.CL2_3` | `cms.vuint32` | Channels for CL2 link 3 (required if `bufferFileType` = `input`) |
| `OutputChannels.GTTPromptJets` | `cms.vuint32` | Channels for collection GTTPromptJets (required if `bufferFileType` = `output`) |
| `OutputChannels.GTTDisplacedJets` | `cms.vuint32` | Channels for collection GTTDisplacedJets (required if `bufferFileType` = `output`) |
| `OutputChannels.GTTPromptHtSum` | `cms.vuint32` | Channels for collection GTTPromptHtSum (required if `bufferFileType` = `output`) |
| `OutputChannels.GTTDisplacedHtSum` | `cms.vuint32` | Channels for collection GTTDisplacedHtSum (required if `bufferFileType` = `output`) |
| `OutputChannels.GTTEtSum` | `cms.vuint32` | Channels for collection GTTEtSum (required if `bufferFileType` = `output`) |
| `OutputChannels.GTTPrimaryVert` | `cms.vuint32` | Channels for collection GTTPrimaryVert (required if `bufferFileType` = `output`) |
| `OutputChannels.GMTSaPromptMuons` | `cms.vuint32` | Channels for collection GMTSaPromptMuons (required if `bufferFileType` = `output`) |
| `OutputChannels.GMTSaDisplacedMuons` | `cms.vuint32` | Channels for collection GMTSaDisplacedMuons (required if `bufferFileType` = `output`) |
| `OutputChannels.GMTTkMuons` | `cms.vuint32` | Channels for collection GMTTkMuons (required if `bufferFileType` = `output`) |
| `OutputChannels.CL2JetsSC4` | `cms.vuint32` | Channels for collection CL2JetsSC4 (required if `bufferFileType` = `output`) |
| `OutputChannels.CL2JetsSC8` | `cms.vuint32` | Channels for collection CL2JetsSC8 (required if `bufferFileType` = `output`) |
| `OutputChannels.CL2Photons` | `cms.vuint32` | Channels for collection CL2Photons (required if `bufferFileType` = `output`) |
| `OutputChannels.CL2Electrons` | `cms.vuint32` | Channels for collection CL2Electrons (required if `bufferFileType` = `output`) |
| `OutputChannels.CL2Taus` | `cms.vuint32` | Channels for collection CL2Taus (required if `bufferFileType` = `output`) |
| `OutputChannels.CL2EtSum` | `cms.vuint32` | Channels for collection CL2EtSum (required if `bufferFileType` = `output`) |
| `OutputChannels.CL2HtSum` | `cms.vuint32` | Channels for collection CL2HtSum (required if `bufferFileType` = `output`) |

Note: In order to get consistency across multiple pattern files written by multiple writers it is recommended to produce patterns in single threaded mode only (i.e. `process.options.numberOfThreads = 1`).

Default configurations for `L1GTAlgoBoardWriter` and `L1GTObjectBoardWriter` in input and output direction can be pulled into the configuration for each of the two prototype implementations VU9P and VU13P via:

```python
# Serenity VU9P prototype board
process.load('L1Trigger.Phase2L1GT.l1tGTBoardWriterVU9P_cff')

process.pBoardDataInputVU9P = cms.EndPath(process.BoardDataInputVU9P)
process.pBoardDataOutputObjectsVU9P = cms.EndPath(process.BoardDataOutputObjectsVU9P)
process.pAlgoBitBoardDataVU9P = cms.EndPath(process.AlgoBitBoardDataVU9P)
```

```python
# Serenity VU13P prototype board
process.load('L1Trigger.Phase2L1GT.l1tGTBoardWriterVU13P_cff')

process.pBoardDataInputVU13P = cms.EndPath(process.BoardDataInputVU13P)
process.pBoardDataOutputObjectsVU13P = cms.EndPath(process.BoardDataOutputObjectsVU13P)
process.pAlgoBitBoardDataVU13P = cms.EndPath(process.AlgoBitBoardDataVU13P)
```