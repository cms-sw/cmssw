# Generate and simulate a single electron gun sample, with a limited simulation geometry
cmsDriver.py SingleElectronPt10_pythia8_cfi  --conditions auto:run2_mc -n 10 --eventcontent FEVTDEBUG --relval 9000,3000 -s\
 GEN,SIM --datatier GEN-SIM --beamspot NominalCollision2015 --customise SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1 --magField 38T_PostLS1 --geometry ECALHCAL --fileout gensim.root

# Standard digitisation with the standard geometry
# Use customisation functions to put empty SimHit collections for the detector parts that were not simulated
cmsDriver.py step2  --customise SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1 --customise FastSimulation/Configuration/Customs.fakeSimHits_for_geometry_ECALHCAL --conditions auto:run2_mc -s DIGI:pdigi_valid,L1,DIGI2\
RAW,HLT:@relval25ns,RAW2DIGI,L1Reco --datatier GEN-SIM-DIGI-RAW-HLTDEBUG -n 10 --magField 38T_PostLS1 --eventcontent FEVTDEBUGHLT --filein file:gensim.root --fileout digi.root

# Reconstruct with the standard reconstruction sequence
cmsDriver.py step3  --customise SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1 --conditions auto:run2_mc -s RAW2DIGI,L1Reco,RECO,EI,V\
ALIDATION,DQM --datatier GEN-SIM-RECO,DQMIO -n 10 --magField 38T_PostLS1 --eventcontent FEVTDEBUGHLT,DQM --filein file:digi.root --fileout reco.root
