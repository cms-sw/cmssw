#cp ../../ComputeGains/Data/Gains.root .
root -l -b -q MakeMap.C+
rm *.db
cmsRun MakeMap_Merge_cfg.py
cd Macro
root -l -b -q makePlot.C++
cd ..
