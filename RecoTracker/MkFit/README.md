# mkFit

This package holds the glue modules for running
[mkFit](http://trackreco.github.io/) within CMSSW.

Note that at the moment there may be only one `MkFitProducer` in a
single job. This restriction will be removed in the future.

Also note that at the moment the mkFit works only with the CMS phase1
tracker detector. Support for the phase2 tracker will be added later.

## Modifier for runTheMatrix workflows (offline reconstruction)

* `Configuration.ProcessModifiers.trackingMkFit_cff.trackingMkFit`
  * Replaces initialStep track building module with `mkFit`.

## Customize functions for runTheMatrix workflows (offline reconstruction)

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
$ runTheMatrix.py -l <workflow(s)> --apply 2 --command "--procModifiers trackingMkFit --customise RecoTracker/MkFit/customizeInitialStepToMkFit.customizeInitialStepOnly"
```