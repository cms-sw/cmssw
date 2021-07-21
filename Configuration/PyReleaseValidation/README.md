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
* 0.502: Patatrack, pixel only quadruplets, on GPU
* 0.505: Patatrack, pixel only triplets, on CPU
* 0.506: Patatrack, pixel only triplets, on GPU
* 0.511: Patatrack, ECAL only CPU
* 0.512: Patatrack, ECAL only GPU
* 0.521: Patatrack, HCAL only CPU
* 0.522: Patatrack, HCAL only GPU
* 0.6: HE Collapse (old depth segmentation for 2018)
* 0.7: trackingMkFit modifier
* 0.8: BPH Parking (Run-2)
* 0.9: Vector hits
* 0.12: Neutron background
* 0.13: MLPF algorithm
* 0.15: JME NanoAOD
* 0.17: Run-3 deep core seeding for JetCore iteration
* 0.21: Production-like sequence
* 0.24: 0 Tesla (Run-2, Run-3)
* 0.61: `phase2_ecal_devel` era
* 0.91: Track DNN modifier
* 0.97: Premixing stage1
* 0.98: Premixing stage2
* 0.99: Premixing stage1+stage2
* 0.999: 0.99 with Phase-2 premixing with PU50
* 0.9821: Production-like premixing stage2
* 0.9921: Production-like premixing stage1+stage2
* 0.911: DD4Hep reading geometry from XML
* 0.912: DD4Hep reading geometry from the DB
* 0.101: Phase-2 aging, 1000fb-1
* 0.103: Phase-2 aging, 3000fb-1
* 0.9001: Sonic Triton
