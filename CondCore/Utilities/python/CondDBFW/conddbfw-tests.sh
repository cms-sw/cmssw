source /afs/cern.ch/cms/cmsset_default.sh
cd /afs/cern.ch/user/j/jdawes/CMSSW_7_5_2/ 
eval `scram runtime -sh`
python -m CondCore.Utilities.CondDBFW.tests 
