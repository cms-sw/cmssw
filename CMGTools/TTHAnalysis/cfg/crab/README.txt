1) edit ../run_susyMT2.cfg to run over your favorite components (and set splitting -> NJOBS)

2) edit launchall.py to change the CMG-version/tag and the production name

3) run!!!
> voms-proxy-init -voms cms --valid=50:00
Enter GRID pass phrase for this identity: xxxx
> python launchall.py


Notes: 
- debugging: debug option can be set on launchall.py (be smart and choose a single component in the cfg)
- if useAAA=True in launchall.py, xrootd will be use instead of the default eos (from samples.py)
- modify heppy_crab_config.py to run only on your favorite sites
