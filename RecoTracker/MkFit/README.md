# mkFit

This package holds the glue modules for running
[mkFit](http://trackreco.github.io/) within CMSSW.

Note that at the moment there may be only one `MkFitProducer` in a
single job. This restriction will be removed in the future.

## Customize functions for runTheMatrix workflows (offline reconstruction)

* `RecoTracker/MkFit/customizeInitialStepToMkFit.customizeInitialStepToMkFit`
  * Replaces initialStep track building module with `mkFit`.
* `RecoTracker/MkFit/customizeInitialStepOnly.customizeInitialStepOnly`
  * Run only the initialStep tracking. In practice this configuration
    runs the initialStepPreSplitting iteration, but named as
    initialStep. MultiTrackValidator is included, and configured to
    monitor initialStep. Intended to provide the minimal configuration
    for CMSSW tests.
* `RecoTracker/MkFit/customizeInitialStepOnly.customizeInitialStepOnlyNoMTV`
  * Otherwise same as `customizeInitialStepOnly` except drops
    MultiTrackValidator. Intended for profiling.


These can be used with e.g.
```bash
$ runTheMatrix.py -l <workflow(s)> --apply 2 --command "--customise RecoTracker/MkFit/customizeInitialStepToMkFit.customizeInitialStepToMkFit"
```