#!/bin/sh

#This script is expected to be executed in the CMSSW_X/src/GeneratorInterface/RivetInterface/test directory

curl -s https://raw2.github.com/cms-sw/genproductions/master/python/SevenTeV/QCD_Pt_470to600_Tune4C_7TeV_pythia8_cff.py -o ../../../Configuration/GenProduction/python/SevenTeV/QCD_Pt_470to600_Tune4C_7TeV_pythia8_cff.py --create-dirs

curl -s https://raw2.github.com/cms-sw/genproductions/master/python/rivet_customize.py -o ../../../Configuration/GenProduction/python/rivet_customize.py

cd ../../../

scram b

#go to GeneratorInterface/RivetInterface/test/ subdirectory
cd GeneratorInterface/RivetInterface/test

#do a test run over 10k QCD events with Pythia8 and fill histograms for the analysis CMS_2013_I1224539_DIJET
cmsRun rivet_cfg.py

sleep 10

echo "Executing 'rivet-mkhtml mcfile.yoda' now!"
#last step: check in browser the generated events for the given CMS analysis
rivet-mkhtml mcfile.yoda


echo "Executing 'firefox plots/index/html' now!"
#point browser to plots/index.html file to check plots
firefox plots/index.html

#you might want to setup the Rivet environment variables via
#rivetSetup.csh or rivetSetup.sh depending on your shell
# (not needed for the provided example analysis)