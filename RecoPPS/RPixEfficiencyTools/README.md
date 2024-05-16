## Onboarding
### Onboarding - how to setup workspace 
1. login to ssh 
```
ssh <username>@lxplus.cern.ch
# example
ssh mobrzut@lxplus.cern.ch
```

2. Go to your afs directory *(note1: HTCondor submission required - currently not available on eos)*
*(note2: check if you have afs enabled and at least 5Gb of storage [cern resources portal](https://resources.web.cern.ch/resources/Manage/ListServices.aspx))*

```
#example
cd /afs/cern.ch/user/m/mobrzut
```
3. It's best to create a separate folder (ex. `mkdir automation`) and setup there a CMSSW. (follow `project setup` section on readme)

### Onboarding - how to browse root files
There are a bunch of methods to browse histograms of root file. 
**Method 1**
1. Connect to ssh with flag -XY
```
ssh -XY <username>@lxplus.cern.ch
```

2. Go to the directory with your file and open there root. 
```
root
``` 
3. Open TBrowser. Now GUI should appear. 
```
new TBrowser
```

4. You can exit from root by quit command  
```.q```

**Method 2**
Install VS code extension [ROOT File Viewer](https://marketplace.visualstudio.com/items?itemName=albertopdrf.root-file-viewer) by *Alberto Perez de Rada Fiol*. 
Then navigate to root file via UI of VS Code and open it. 
***Warning:** This extension might not show all histograms*
 
## Project setup 
Instruction on how to setup and run efficiency suite in CMSSW_12_4_0 environment:

1. Preprare CMSSW_12_4_0 environment with: `cmsrel CMSSW_12_4_0`
2. Go to newly created directory: `cd CMSSW_12_4_0`
3. Setup cms environmental variables(repeat every time you log in): `cmsenv`
4. Merge with the main branch containing efficiency suite: `git cms-merge-topic CTPPS:rolling_calib_efficiency_tracking_12_4_0`
5. Compile the solution `scram b -j10`
6. Switch directory to the one containing the source code: `cd src/RecoPPS/RPixEfficiencyTools`
7. Create directories for files used during the run: `mkdir InputFiles OutputFiles Jobs LogFiles`

### Preparation fo Input File list
8. Prepare the input .dat file for the chosen era with <era name>. For instance, you can specify this file to load all the input .root files from the chosen directory by typing:
	ls /path/to/your/input/files/*.root | sed 's/^/file:/' > InputFiles/Era<era name>.dat


## How to test Efficiency Analyis (referred as EA) and Reference Analysis (referred as RA)
1. Set up CMSSW_12_4_0 project (follow steps 1-7 of `Project Setup` section) just checkout different branch
	1. *(STEP 4 modification)* `git cms-merge-topic CTPPS:MO/automation/for_CMSSW12-update-DQMWorker`
	2. Remember to run `cmsenv` in each terminal session at src directory
	3. Remember to create needed directories:
		```mkdir InputFiles OutputFiles Jobs LogFiles```
2. Prepare test file and test setup:
	1. **file** *(note/TODO: this file is for EA, it works with RA (suitable for simple test)  but the results are not useful from physics perspective)* `/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2018/ReRecoEras_CMSSW_10_6_10/EGamma/re-MINIAOD/ctpps_2018A_pixelefficiency_singleele200630_073503/0000/reMINIAOD_A.root`
	2. **file list to analyse** Just create .dat file with paths you want to analyse.
	```ls /eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2018/ReRecoEras_CMSSW_10_6_10/EGamma/re-MINIAOD/ctpps_2018A_pixelefficiency_singleele200630_073503/0000/*.root | sed 's/^/file:/' > InputFiles/Era.dat```
	3. **Set up config files** (.py files in python folder).  You just need to make step 1 manually, all others values are correct. They are put here for debugging purposes.  
		1. **maxEvents** as test file is big just set maxEvents to smaller value ex. `100000`
		2. **Global tag** Use `123X_dataRun2_v4` Global Tag in the python files.
		3. **Process**  `eras.Run2_2018` (Change according to the period you want to analyse)
		4. **Producer Tag** `CTPPSTestProtonReconstruction`
		
3. Run Efficiency analysis.
	1. **Run EA (efficiency analysis) worker**
		```cmsRun python/EfficiencyAnalysisDQMWorker_cfg.py sourceFileList=InputFiles/Era.dat outputFileName=ea_tmp.root bunchSelection=NoSelection```
	2. **Run EA (efficiency analysis) harvester** To run EA you need to specify 5 paramteres
		1. **inputFileName** .root output file of EA Worker
		2. **outputDirectoryPath** where to save output of root file 
		3. **workflow** ex. efficiency, timing resolution
		4. **dataPeriod** runNumber or lumisection ex. `315512`
		5. **campaign** provides a unique naming for distinction between same workflow ex. `25_07_22`
	```cmsRun python/EfficiencyAnalysisDQMHarvester_cfg.py inputFileName=ea_tmp.root outputDirectoryPath=OutputFiles workflow=randomNameWorkflow dataPeriod=dumbDataPeriod campaign=fooCampaignName```

4. Run Reference Analysis 
	1. **Run RA worker** 
		1. **sourceFileList** self describing
		2. **outputFileName** self describing 
		3. **efficiencyFileName** .root file output of EA Harvester
	```cmsRun python/ReferenceAnalysisDQMWorker_cfg.py sourceFileList=InputFiles/Era.dat outputFileName=tmp.root efficiencyFileName=OutputFiles/DQM_V0001_R000999999__fooCampaignName__randomNameWorkflow__dumbDataPeriod.root```

	2. **Run RA harvester**

	TODO: Probably outputDirectoryPath doesn't change the output path. 
	```cmsRun python/ReferenceAnalysisDQMHarvester_cfg.py inputFileName=tmp.root outputDirectoryPath=OutputFiles```
5. Verification
	1. Open root file from RA Harvester *(note: More instruction in the `Onboarding section` of this readme)*
	2. Check a few histograms. Ex.`h2RefinedTrackEfficiency_am0_*`, `h2TrackHitDistribution_*`
	Below images of example histograms.
	![](ReadmeResources/h2_RefinedTrackEfficiency_example_histogram.png)
	![](ReadmeResources/h2_HitTrackDistribution_example_histogram.png)

	3. You can limit data to `colz` (Set `draw option`) 

# TODO - Hope it will be done in a finite time :)  
1. [ ] Global tag is set in every python file separately - if you change a global tag you need to edit four files!!!
2. [ ] Era modifier is set in every python file separately
3. [ ] Set default parameters for input and output files or directories in config files  

