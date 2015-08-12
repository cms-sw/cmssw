to run the scripts do
source CMSSW_XX/src/L1TriggerDPG/scripts/setup.sh
To run these on the IC batch queue there are several steps:

1) Make sure you have batch access.
2) voms-proxy-init -voms cms
3) run l1analysis.sh and do .L script.C+ to build the shared libaries (this doesnt work from batch). Exit root.
4) Configure subber.py and pyRun.py with the list of samples you wish to run over.
5) Run subber.py with the arguments [0] - Analysis name (with out the .C extension) [1] output directory.
6) wait for the jobs to finish and hadd the outputs.
