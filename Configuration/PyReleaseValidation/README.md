# runTheMatrix.py

## Upgrade workflows

To list the upgrade workflows:
```
runTheMatrix.py --what upgrade -n
```

To make an upgrade workflow visible to the regular matrix, add it to:
* [relval_2017.py](./python/relval_2017.py) (for Run 2 and Run 3)
* [relval_Run4.py](./python/relval_Run4.py) (for Phase 2)

All workflows in the regular matrix can be run in IB tests,
so this should only be done for fully functional workflows.

To add a workflow to the PR tests, modify the `'limited'` list in
[runTheMatrix.py](./scripts/runTheMatrix.py).
(This should be done sparingly to limit the computational burden of PR tests.)

### Special workflow offsets

Special workflows are defined in [upgradeWorkflowComponents.py](./python/upgradeWorkflowComponents.py).
Each workflow must have a unique offset.
The base `UpgradeWorkflow` class can be extended to implement a particular special workflow,
specifying which steps to modify and for which conditions to create the workflow.

The offsets currently in use are:
* 0.1: Tracking-only validation and DQM
* 0.2: Tracking Run-2 era, `Run2_2017_trackingRun2`
* 0.3: 0.1 + 0.2
* 0.4: LowPU tracking era, `Run2_2017_trackingLowPU`
* 0.402: Alpaka, pixel only quadruplets, portable
* 0.403: Alpaka, pixel only quadruplets, portable vs. CPU validation
* 0.404: Alpaka, pixel only quadruplets, portable profiling
* 0.406: Alpaka, pixel only triplets, portable
* 0.407: Alpaka, pixel only triplets, portable vs. CPU validation
* 0.407: Alpaka, pixel only triplets, portable profiling
* 0.412: Alpaka, ECAL only, portable
* 0.413: Alpaka, ECAL only, portable vs. CPU validation
* 0.422: Alpaka, HCAL only, portable
* 0.423: Alpaka, HCAL only, portable vs CPU validation
* 0.424: Alpaka, HCAL only, portable profiling
* 0.492: Alpaka, full reco with pixel quadruplets
* 0.496: Alpaka, full reco with pixel triplets
* 0.5: Legacy pixel tracking only (CPU)
* 0.511: Legacy ECAL reco only (CPU)
* 0.521: Legacy HCAL reco only (CPU)
* 0.6: HE Collapse (old depth segmentation for 2018)
* 0.601: HLT as separate step
* 0.7: trackingMkFit modifier
* 0.701: DisplacedRegionalStep tracking iteration for Run-3
* 0.702: trackingMkFit modifier for Phase-2 (initialStep only)
* 0.703: LST tracking (Phase-2 only), initialStep+HighPtTripletStep only, on CPU
* 0.704: LST tracking (Phase-2 only), initialStep+HighPtTripletStep only, on GPU (if available)
* 0.75: HLT phase-2 timing menu
* 0.751: HLT phase-2 timing menu Alpaka variant
* 0.752: HLT phase-2 timing menu ticl_v5 variant
* 0.753: HLT phase-2 timing menu Alpaka, single tracking iteration variant
* 0.754: HLT phase-2 timing menu Alpaka, single tracking iteration, LST building variant
* 0.755: HLT phase-2 timing menu Alpaka, LST building variant
* 0.756 HLT phase-2 timing menu trimmed tracking
* 0.7561 HLT phase-2 timing menu Alpaka, trimmed tracking
* 0.7562 HLT phase-2 timing menu Alpaka, trimmed tracking, single tracking iteration variant
* 0.7563 HLT phase-2 timing menu trimmed tracking, LST building variant
* 0.7664 HLT phase-2 timing menu Alpaka, trimmed tracking, LST building variant
* 0.7665 HLT phase-2 timing menu Alpaka, trimmed tracking, single tracking iteration variant, LST building variant
* 0.78: Complete L1 workflow
* 0.8: BPH Parking (Run-2)
* 0.81: Running also HeavyFlavor DQM
* 0.85: Phase-2 Heavy Ion
* 0.9: Vector hits
* 0.12: Neutron background
* 0.13: MLPF algorithm
* 0.15: JME NanoAOD
* 0.17: Run-3 deep core seeding for JetCore iteration
* 0.19: ECAL SuperClustering with DeepSC algorithm
* 0.21: Production-like sequence
* 0.21X1 : Production-like sequence with classical mixing PU=X (X=10,20,30,40,50,60,70,80,90,100,120,140,160,180)
* 0.24: 0 Tesla (Run-2, Run-3)
* 0.31: Photon energy corrections with DRN architecture
* 0.61: ECAL `phase2_ecal_devel` era, on CPU
* 0.612: ECAL `phase2_ecal_devel` era, with automatic offload to GPU if available
* 0.631: ECAL component-method based digis
* 0.632: ECAL component-method based finely-sampled waveforms
* 0.633: ECAL phase2 Trigger Primitive
* 0.634: ECAL phase2 Trigger Primitive + component-method based digis
* 0.635: ECAL phase2 Trigger Primitive + component-method based finely-sampled waveforms
* 0.91: Track DNN modifier
* 0.97: Premixing stage1
* 0.98: Premixing stage2
* 0.99: Premixing stage1+stage2
* 0.999: 0.99 with Phase-2 premixing with PU50
* 0.9821: Production-like premixing stage2
* 0.9921: Production-like premixing stage1+stage2
* 0.911: DD4hep reading geometry from XML
* 0.912: DD4hep reading geometry from the DB
* 0.914: DDD DB
* 0.101: Phase-2 aging, 1000fb-1
* 0.103: Phase-2 aging, 3000fb-1
* 0.201: HGCAL special TICL Pattern recognition Workflows: clue3D
* 0.202: HGCAL special TICL Pattern recognition Workflows: FastJet
* 0.203: HGCAL TICLv5
* 0.204: HGCAL superclustering : using Mustache in TICLv5
* 0.205: HGCAL superclustering : using old PFCluster-based Mustache algorithm with TICLv5
* 0.302: FastSim Run-3 trackingOnly validation
* 0.303: FastSim Run-3 MB for mixing
* 0.9001: Sonic Triton
* 0.278: Weighted Vertexing in Blocks
* 0.279: Weighted Vertexing in Blocks and tracking only wf
* 0.111: Activate OuterTracker inefficiency (PS-p: bias rails inefficiency only)
* 0.112: Activate OuterTracker inefficiency (PS-p: bias rails inefficiency; PS-s and SS: 1% bad strips)
* 0.113: Activate OuterTracker inefficiency (PS-p: bias rails inefficiency; PS-s and SS: 5% bad strips)
* 0.114: Activate OuterTracker inefficiency (PS-p: bias rails inefficiency; PS-s and SS: 10% bad strips)
* 0.141: Activate emulation of the signal shape of the InnerTracker FE chip (CROC)
