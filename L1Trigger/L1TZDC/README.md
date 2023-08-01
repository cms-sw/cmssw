Below README for testing prior to CMSSW integration


***** COMMON FIXES ON FAIL (Skip to below for build instructions) *****

If you see output like the following
```
----- Begin Fatal Exception 18-Aug-2023 22:34:50 CEST-----------------------
An exception of category 'ProductNotFound' occurred while
   [0] Processing  Event run: 326295 lumi: 4 event: 32601 stream: 0
   [1] Running path 'zdcEtSum'
   [2] Calling method for module L1TZDCProducer/'zdcEtSumProducer'
Exception Message:
Principal::getByToken: Found zero products matching all criteria
Looking for type: HcalDataFrameContainer<QIE10DataFrame>
Looking for module label: hcalDigis
Looking for productInstanceName: ZDC
Looking for process: reRECO
   Additional Info:
      [a] If you wish to continue processing events after a ProductNotFound exception,
add "SkipEvent = cms.untracked.vstring('ProductNotFound')" to the "options" PSet in the configuration.

----- End Fatal Exception -------------------------------------------------
```

The issue is almost certainly the inclusion of "reRECO" as a third argument for the zdcDigis input tag to zDCEtSumProducer (CM note: putting this at the top as I have forgotten this issue twice now)

***********************************************************************


Basic build instructions, integratable into Molly's L1Emulator instructions with the Run 3 HI menu using CMSSW_13_1_0_pre4 Found here: \
https://github.com/mitaylor/L1StudiesFramework/tree/main/RunPbPbL1Ntuples

To build, do
```
cmsrel CMSSW_13_1_0_pre4
cd CMSSW_13_1_0_pre4/src
cmsenv
git cms-init
#Insert zdcL1T_v0.0.X
git remote add cfmcginn https://github.com/cfmcginn/cmssw.git
git fetch cfmcginn zdcL1TOnCMSSW_13_1_0_pre4
git cms-merge-topic -u cfmcginn:zdcL1T_latest
#Note we will do the next line using https instead of Molly's ssh instructions
#git remote add cms-l1t-offline git@github.com:cms-l1t-offline/cmssw.git
git remote add cms-l1t-offline https://github.com/cms-l1t-offline/cmssw.git
git fetch cms-l1t-offline l1t-integration-CMSSW_13_1_0_pre4
git cms-merge-topic -u cms-l1t-offline:l1t-integration-v161
git clone https://github.com/cms-l1t-offline/L1Trigger-L1TCalorimeter.git L1Trigger/L1TCalorimeter/data
svn export https://github.com/boundino/HltL1Run2021.git/trunk/L1/ADC

git cms-checkdeps -A -a

scram b -j 8

wget https://raw.githubusercontent.com/ginnocen/UPCopenHFanalysis/main/zdc_calibration/newZDCAnalyzer/test/files_327524.txt
mv files_327524.txt L1Trigger/L1TZDC/test/
```

To test, do
```
cd L1Trigger/L1TZDC/test
cmsRun l1ZDCProducerTest.py
```

Continuing, but now explicitly using Molly's build instructions directly (Step 2)

```
git cms-addpkg L1Trigger/L1TCommon
git cms-addpkg L1Trigger/L1TGlobal
mkdir -p L1Trigger/L1TGlobal/data/Luminosity/startup/
cd L1Trigger/L1TGlobal/data/Luminosity/startup/
wget https://raw.githubusercontent.com/mitaylor/HIMenus/main/Menus/L1Menu_CollisionsHeavyIons2023_v0_0_1.xml
cd ../../../../../
scram b -j 8
```
On a good build we need to edit customiseUtils.py per Molly's instructions:

emacs -nw L1Trigger/Configuration/python/customiseUtils.py

process.TriggerMenu.L1TriggerMenuFile = cms.string('L1Menu_Collisions2022_v1_2_0.xml') → process.TriggerMenu.L1TriggerMenuFile = cms.string('L1Menu_CollisionsHeavyIons2023_v0_0_1.xml')

Create the python by grabbing Molly's runCmsDriver for 2018 data
```
wget https://raw.githubusercontent.com/mitaylor/L1StudiesFramework/main/RunPbPbL1Ntuples/runCmsDriver_2018Data.sh
bash runCmsDriver_2018Data.sh
```

We need to modify the output, l1Ntuple_2018Data.py
Towards the end add this block, but before the line

"MassReplaceInputTag(process, new="rawDataMapperByLabel", old="rawDataCollector")"
****************************
```
process.l1UpgradeTree.sumZDCToken = cms.untracked.InputTag("zdcEtSumProducer")

process.l1UpgradeEmuTree.sumZDCToken = cms.untracked.InputTag("zdcEtSumProducer")

process.zdcEtSumProducer = cms.EDProducer('L1TZDCProducer',
  zdcDigis = cms.InputTag("hcalDigis", "ZDC")
)

process.zdcEtSum = cms.Path(process.zdcEtSumProducer)
process.schedule.append(process.zdcEtSum)

#ABOVE CODE BEFORE THIS LINE
MassReplaceInputTag(process, new="rawDataMapperByLabel", old="rawDataCollector")
```
****************************


This should run out of the box - if it does not please contact me (cfmcginn) or ginnocen @ github