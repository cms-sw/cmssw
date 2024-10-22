# mkFit

This package holds the glue modules for running
[mkFit](http://trackreco.github.io/) within CMSSW.

Note that at the moment there may be only one `MkFitProducer` in a
single job. This restriction will be removed in the future.

Also note that at the moment the mkFit works only with the CMS phase1
tracker detector. Support for the phase2 tracker will be added later.

## Modifier for runTheMatrix workflows (offline reconstruction)

* `Configuration/Eras/python/ModifierChain_trackingMkFitProd_cff.py`
  * Replaces track building module with `mkFit` for 6 tracking iterations: 
     * InitialStepPreSplitting
     * InitialStep
     * HighPtTripletStep
     * DetachedQuadStep
     * DetachedTripletStep
     * PixelLessStep

* `Configuration/ProcessModifiers/python/trackingMkFitDevel_cff.py`
  * Replaces track building module with `mkFit` for all tracking iterations

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
$ runTheMatrix.py -l <workflow(s)> --apply 2 --command "--procModifiers trackingMkFitDevel --customise RecoTracker/MkFit/customizeInitialStepToMkFit.customizeInitialStepOnly"
```

## Description of configuration parameters

### Iteration configuration [class IterationConfig]

* *m_track_algorithm:* CMSSW track algorithm (used internally for reporting and consistency checks)
* *m_requires_seed_hit_sorting:* do hits on seed tracks need to be sorted (required for seeds that include strip layers)
* *m_params:* IterationParams structure for this iteration
* *m_backward_params:* IterationParams structure for backward search for this iteration
* *m_layer_configs:* std::vector of per-layer parameters

#### Seed cleaning params (based on elliptical dR-dz cut) [in class IterationConfig]

* *m_seed_cleaner_name:* name of standard function to call for seed cleaning; if not set or empty seed cleaning is not performed
* *sc_ptthr_hpt:* pT threshold used to tighten seed cleaning requirements
* *sc_drmax_bh:* dR cut used for seed tracks with std::fabs(eta)<0.9 and pT > c_ptthr_hpt
* *sc_dzmax_bh:* dz cut used for seed tracks with std::fabs(eta)<0.9 and pT > c_ptthr_hpt
* *sc_drmax_eh:* dR cut used for seed tracks with std::fabs(eta)>0.9 and pT > c_ptthr_hpt
* *sc_dzmax_eh:* dz cut used for seed tracks with std::fabs(eta)>0.9 and pT > c_ptthr_hpt
* *sc_drmax_bl:* dR cut used for seed tracks with std::fabs(eta)<0.9 and pT < c_ptthr_hpt
* *sc_dzmax_bl:* dz cut used for seed tracks with std::fabs(eta)<0.9 and pT < c_ptthr_hpt
* *sc_drmax_el:* dR cut used for seed tracks with std::fabs(eta)>0.9 and pT < c_ptthr_hpt
* *sc_dzmax_el:* dz cut used for seed tracks with std::fabs(eta)>0.9 and pT < c_ptthr_hpt

#### Seed partitioning params [in class IterationConfig]

* *m_seed_partitioner_name:* name of standard function to call for seed partitioning

#### Pre / post backward-fit candidate top-level params [in class IterationConfig]

* *m_pre_bkfit_filter_name:* name of standard function used for candidate filtering after forward
search but before backward fit; if not set or empty no candidate filtering is performed at this stage
* *m_post_bkfit_filter_name:* name of standard function used for candidate filtering after backward fit; if not set or empty no candidate filtering is performed at this stage

#### Duplicate cleaning parameters [in class IterationConfig]

* *m_duplicate_cleaner_name:* name of standard function used for duplicate cleaning; if not set or empty duplicate cleaning is not performed
* *dc_fracSharedHits:* min fraction of shared hits to determine duplicate track candidate
* *dc_drth_central:* dR cut used to identify duplicate candidates if std::abs(cotan(theta))<1.99 (abs(eta)<1.44)
* *dc_drth_obarrel:* dR cut used to identify duplicate candidates if 1.99<std::abs(cotan(theta))<6.05 (1.44<abs(eta)<2.5)
* *dc_drth_forward:* dR cut used to identify duplicate candidates if std::abs(cotan(theta))>6.05 (abs(eta)>2.5)

### Iteration parameters [class IterationParams]

* *nlayers_per_seed:* internal mkFit parameter used for standalone validation
* *maxCandsPerSeed:* maximum number of concurrent track candidates per given seed
* *maxHolesPerCand:* maximum number of allowed holes on a candidate
* *maxConsecHoles:*  maximum number of allowed consecutive holes on a candidate
* *chi2Cut_min:*     minimum chi2 cut for accepting a new hit
* *chi2CutOverlap:*  chi2 cut for accepting an overlap hit (currently NOT used)
* *pTCutOverlap:*    pT cut below which the overlap hits are not picked up

#### Pre / post backward-fit candidate filtering params
* *minHitsQF:* minimum number of hits, interpretation depends on particular filtering function used

### Per-layer parameters [class IterationLayerConfig]

* *m_select_min_dphi, m_select_max_dphi:* geometry-driven dphi baseline selection window cut
* *m_select_min_dq, m_select_max_dq:* geometry-driven dr (endcap) / dz (barrel) baseline selection window cut
* *c_dp_[012]:* dphi selection window cut (= [0]*1/pT + [1]*std::fabs(theta-pi/2) + [2])
* *c_dp_sf:* additional scaling factor for dphi cut
* *c_dq_[012]:* dr (endcap) / dz (barrel) selection window cut (= [0]*1/pT + [1]*std::fabs(theta-pi/2) + [2])
* *c_dq_sf:* additional scaling factor for dr (endcap) / dz (barrel) cut
* *c_c2_[012]:* chi2 cut for accepting new hit (= [0]*1/pT + [1]*std::fabs(theta-pi/2) + [2])
* *c_c2_sf:* additional scaling factor for chi2 cut
