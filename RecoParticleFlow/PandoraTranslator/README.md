# PandoraTranslator
CMS -> PandoraPFA -> CMS Particle Flow Translator

## instructions on lxplus or cmslpc

cmsrel CMSSW_6_2_0_SLHC23_patch2

cd CMSSW_6_2_0_SLHC23_patch2/src && cmsenv

git cms-init

git clone git@github.com:lgray/PandoraTranslator.git ./HGCal/PandoraTranslator

source HGCal/PandoraTranslator/scripts/buildPandoraInstallTool.sh

git clone https://github.com/sethzenz/HGCanalysis.git --branch hacked-interactions-filter UserCode/HGCanalysis

scram b -j 9
