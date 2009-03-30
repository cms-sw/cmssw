#!/bin/sh

cmsRun DummyCondDBWriter_SiStripApvGain_cfg.py 

cat DummyCondDBWriter_SiStripApvGain_cfg.py | sed -e "s@SiStripApvGain_Ideal@SiStripApvGain_IdealSim@" > DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py

cmsRun DummyCondDBWriter_SiStripBadChannel_cfg.py
cmsRun DummyCondDBWriter_SiStripBadFiber_cfg.py
cmsRun DummyCondDBWriter_SiStripBadModule_cfg.py

cmsRun DummyCondDBWriter_SiStripThreshold_cfg.py
cmsRun DummyCondDBWriter_SiStripClusterThreshold_cfg.py

cmsRun DummyCondDBWriter_SiStripFedCabling_cfg.py

cmsRun DummyCondDBWriter_SiStripLorentzAngle_cfg.py
cat DummyCondDBWriter_SiStripLorentzAngle_cfg.py | sed -e "s@SiStripLorentzAngle_Ideal@SiStripLorentzAngle_IdealSim@" -e s"@PerCent_Err=20.@PerCent_Err=0.@"> DummyCondDBWriter_tmp_cfg.py
cmsRun DummyCondDBWriter_tmp_cfg.py

cmsRun DummyCondDBWriter_SiStripDetVOff_cfg.py

cmsRun DummyCondDBWriter_SiStripNoises_cfg.py
cmsRun DummyCondDBWriter_SiStripNoises_PeakMode_cfg.py

cmsRun DummyCondDBWriter_SiStripPedestals_cfg.py

cmsRun 
cmsRun 
