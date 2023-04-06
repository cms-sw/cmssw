# runTheMatrix.py

## Upgrade workflows

To list the upgrade workflows:
```
runTheMatrix.py --what upgrade -n
```

To make an upgrade workflow visible to the regular matrix, add it to:
* [relval_2017.py](./python/relval_2017.py) (for Run 2 and Run 3)
* [relval_2026.py](./python/relval_2026.py) (for Phase 2)

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
* 0.5: Pixel tracking only + 0.1
* 0.501: Patatrack, pixel only quadruplets, on CPU
* 0.502: Patatrack, pixel only quadruplets, with automatic offload to GPU if available
* 0.504: Patatrack, pixel only quadruplets, GPU profiling
* 0.505: Patatrack, pixel only triplets, on CPU
* 0.506: Patatrack, pixel only triplets, with automatic offload to GPU if available
* 0.508: Patatrack, pixel only triplets, GPU profiling
* 0.511: Patatrack, ECAL only, on CPU
* 0.512: Patatrack, ECAL only, with automatic offload to GPU if available
* 0.513: Patatrack, ECAL only, GPU vs. CPU validation
* 0.514: Patatrack, ECAL only, GPU profiling
* 0.521: Patatrack, HCAL only, on CPU
* 0.522: Patatrack, HCAL only, with automatic offload to GPU if available
* 0.524: Patatrack, HCAL only, GPU profiling
* 0.591: Patatrack, full reco with pixel quadruplets, on CPU
* 0.592: Patatrack, full reco with pixel quadruplets, with automatic offload to GPU if available
* 0.595: Patatrack, full reco with pixel triplets, on CPU
* 0.596: Patatrack, full reco with pixel triplets, with automatic offload to GPU if available
* 0.6: HE Collapse (old depth segmentation for 2018)
* 0.601: HLT as separate step
* 0.7: trackingMkFit modifier
* 0.8: BPH Parking (Run-2)
* 0.81: Running also HeavyFlavor DQM
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
* 0.75: Phase-2 HLT
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
* 0.302: FastSim Run-3 trackingOnly validation
* 0.303: FastSim Run-3 MB for mixing
* 0.9001: Sonic Triton
* 0.278: Weighted Vertexing in Blocks
* 0.279: Weighted Vertexing in Blocks and tracking only wf

