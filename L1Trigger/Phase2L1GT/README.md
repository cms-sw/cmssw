# Menu configuration manual

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
        minQualityScore = cms.uint32(0)
    ),
    collection2 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minPt = cms.double(3),
        maxAbsEta = cms.double(2.4),
        minQualityScore = cms.uint32(0)
    ),
    collection3 = cms.PSet(
        tag = cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        minPt = cms.double(3),
        maxAbsEta = cms.double(2.4),
        minQualityScore = cms.uint32(0)
    ),
    # Correlations are ambiguous (can be {1,2}, {1,3}, or {2,3}), correlXY PSets are thus required.
    correl12 = cms.PSet(
        maxDz = cms.double(1)
    ),
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

\* : To select a $Z_0$ index $i$ from the `GTTPrimaryVert` collection for the comparison use `primVertex = cms.uint32(i)`. This parameter is mandatory when using a `maxPrimVertDz` cut.

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
