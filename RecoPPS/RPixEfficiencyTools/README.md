Instruction on how to setup and run efficiency suite in CMSSW_11_3_2 environment:
1. Preprare CMSSW_11_3_2 environment with: cmsrel CMSSW_11_3_2
2. Go to newly created directory: cd CMSSW_11_3_2
3. Setup cms environmental variables: cmsenv
4. Merge with the topic containing efficiency suite: git cms-merge-topic varsill:from-CMSSW_11_3_2
5. Compile the solution: scram b -j10
6. Switch directory to the one containing the source code: cd src/RecoPPS/RPixEfficiencyTools
7. Create directories for files used during the run: mkdir InputFiles OutputFiles Jobs LogFiles
8. Prepare the input .dat file for the chosen era with <era name>. For instance, you can specify this file to load all the input .root files from the chosen directory by typing:
	ls /path/to/your/input/files/*.root | sed 's/^/file:/' > InputFiles/Era<era name>.dat
9. bash submitEfficiencyAnalysisEra.sh <era name>