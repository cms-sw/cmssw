# SiPixelPhase1ClustersFiltered

In the case that one wishes to create `SiPixelPhaseI` style DQM plots with additional [`GenericTriggerEventFlag`s filtering](https://github.com/cms-sw/cmssw/blob/CMSSW_9_0_X/DQM/SiStripMonitorCluster/src/SiStripMonitorCluster.cc#L55) we attempt to keep the syntax as similar to the original `SiPixelPhaseI` as possible, both in the C++ and python parts

---

## How to make your own plotter
Here we are going to use the [`SiPixelPhase1ClustersFiltered` class](src/SiPixelPhase1ClustersFiltered.cc) class a example for creating your own plotter analyzer class. In the declaration:

```c++
#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1FlagBase.h"
class SiPixelPhase1ClustersFiltered : public SiPixelPhase1FlagBase {
  enum {
    NCLUSTERS
  };

  public:
  explicit SiPixelPhase1ClustersFiltered(const edm::ParameterSet& conf);
  void flagAnalyze(const edm::Event&, const edm::EventSetup&);

  private:
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > srcToken_;
};
```

There are two changes we need to take note here:

1. We are using the [`SiPixelPhase1FlagBase`](../SiPixelPhase1Common/interface/SiPixelPhase1FlagBase.h) base class rather than the [`SiPixelPhase1Base` base class](../SiPixelPhase1Common/interface/SiPixelPhase1Base.h).
2. We are not going to directly use the `analyzer()` method function but rather the `flagAnalyze()` method function. In the implementation, treat the `flagAnalyze()` as you would a `analyzer()` function **without any filters**. The filter processes will be handled elsewhere.

Note that the histogram definition is entirely the same as if you are using a regular [`SiPixelPhase1Base` base class](../SiPixelPhase1Common/README.md).

For the example [python file](python/SiPixelPhase1ClustersFiltered_cfi.py):
```python
SiPixelPhase1ClustersFilteredNClusters
= clusterset.SiPixelPhase1ClustersNClusters.clone (
  name = "filtered_clusters",
  title = "Filtered Clusters",
)

SiPixelPhase1ClustersFilteredConf = cms.VPSet(
  SiPixelPhase1ClustersFilteredNClusters,
)

SiPixelPhase1ClustersFilteredHarvester = cms.EDAnalyzer("SiPixelPhase1Harvester",
    histograms = SiPixelPhase1ClustersFilteredConf,
    geometry = SiPixelPhase1Geometry
)

SiPixelPhase1ClustersFilteredAnalyzer = cms.EDAnalyzer(
    "SiPixelPhase1ClustersFiltered",
    src = cms.InputTag("siPixelClusters"),
    histograms = SiPixelPhase1ClustersFilteredConf,
    geometry = SiPixelPhase1Geometry,

    # Adding filtering
    flaglist = cms.VPSet(
        genericTriggerEventFlag4L1bd,
        genericTriggerEventFlag4HLTdb.
    )
)
```
Notice that the python configuration part for defining the histogram style/dimension is entirely the same as regular [`SiPixelPhase1Base` classes](../SiPixelPhase1Common/python/HistogramManager_cfi.py). The only snippets we need to add is the a `flaglist` parameter set in the final `EDAnalyzer` constructor settings, listing all the `GenericTriggerEventFlag`s we wish to impose. Currently the `genericTriggerEventFlag4L1bd` and `genericTriggerEventFlag4HLTdb` you can see is listed [here](../SiPixelPhase1Common/python/TriggerEventFlag_cfi.py).

Finally, don't forget to add the new `_cfi.py` files to the master `_cfg.py` files you are using. See, [`../SiPixelPhase1Config/python/*_cff.py`](../SiPixelPhase1Config/python) for more examples.

----

## How the code works.

In the implementation of the [`SiPixelPhase1FlagBase` base class](../SiPixelPhase1Common/src/SiPixelPhase1FlagBase.cc), we store the input flag list as a list of `GenericTriggerEventFlag` objects. The raw `analyze()` function in then overloaded to filter according to the `GenericTriggerEventFlag`s given. Only if all the selections is passed will the pure virtual `flagAnalyze()` function that is defined by the users be called.
