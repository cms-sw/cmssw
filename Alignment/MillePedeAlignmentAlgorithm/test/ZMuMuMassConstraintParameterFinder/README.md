# ZMuMuMassConstraintParameterFinder

To obtain the 'Z → µµ' mass constraint parameters do the following:

- modify the value of `dataset` in `submit_jobs.sh`, if needed
- modify cuts in `zmumudistribution_cfg.py`, if needed (see possible cuts in `fillDescriptions` in [`Alignment/MillePedeAlignmentAlgorithm/plugins/ZMuMuMassConstraintParameterFinder.cc`](https://github.com/cms-sw/cmssw/blob/master/Alignment/MillePedeAlignmentAlgorithm/plugins/ZMuMuMassConstraintParameterFinder.cc))
- execute `./submit_jobs.sh`
- when all jobs are finished you will find the parameters in `submit_${dataset}/zMuMuMassConstraintParameters.txt`
-- dummy change --
