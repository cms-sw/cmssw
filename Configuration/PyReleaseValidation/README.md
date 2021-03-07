# runTheMatrix.py

## Upgrade workflows
To list the upgrade workflows,
```
runTheMatrix.py --what upgrade -n
``` 

### offset:
* 0.1: Tracking-only validation and DQM
* 0.2: Tracking Run-2 era, `Run2_2017_trackingRun2`
* 0.3: 0.1 + 0.2
* 0.4: LowPU tracking era, `Run2_2017_trackingLowPU`
* 0.5: Pixel tracking only + 0.1
* 0.6: HE Collapse, HEM in 2018
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
* 0.97, 0.98, 0.99: Premixing stage1, stage2, stage1+stage2
* 0.999: 0.99 with Phase-2 premixing with PU50
* 0.911: DD4Hep
* 0.101: Phase-2 aging, 1000fb-1
* 0.103: Phase-2 aging, 3000fb-1
* 0.501: Patatrack, pixel only CPU
* 0.502: Patatrack, pixel only GPU
* 0.511: Patatrack, ECAL only CPU
* 0.512: Patatrack, ECAL only GPU
* 0.521: Patatrack, HCAL only CPU
* 0.522: Patatrack, HCAL only GPU
* 0.9001: Sonic Triton

