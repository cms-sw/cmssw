
#############################
###  EMTF emulator tools  ###
#############################

-------------------------------------------------
-- Primitive Conversion Look-Up Table generation
-------------------------------------------------

'PC LUTs' are responsible for converting the CSC LCT strip and wire info into phi and theta coordinates
New PC LUTs can be generated for data (using real CMS geometry) or MC as follows
The latest Global Tag can be found at: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideFrontierConditions

cd CMSSW_X_Y_Z/src
cmsenv
cd L1Trigger/L1TMuonEndCap/test/tools/
mkdir -p pc_luts/firmware_data  ## To store the 924 actual LUTs needed by the firmware
mkdir pc_luts/firmware_MC       ## These files are then simplified into 6 text files used by the emulator
cmsRun make_coordlut_data.py    ## To create 924 LUTs from real CMSSW geometry, specified by process.GlobalTag
cmsRun make_coordlut_MC.py      ## Instead uses ideal CMS geometry / alignment from MC
python write_ph_lut_v2.py       ## Modify 'path' in file to specify data or MC

The 6 text files for the emulator will appear in pc_luts/emulator_data or pc_luts/emulator_MC
These can be copied over to the L1Trigger/L1TMuon/data/emtf_luts directory as follows:

cd CMSSW_X_Y_Z/src
git cms-addpkg L1Trigger/L1TMuon
git clone https://github.com/cms-l1t-offline/L1Trigger-L1TMuon.git L1Trigger/L1TMuon/data
mkdir L1Trigger/L1TMuon/data/emtf_luts/ph_lut_new
cp L1Trigger/L1TMuonEndCap/test/tools/pc_luts/emulator_data/* L1Trigger/L1TMuon/data/emtf_luts/ph_lut_new/

The new path can then be added to L1Trigger/L1TMuonEndCap/src/SectorProcessorLUT.cc in the lines containing "ph_lut"

To validate that the coordinate transformation worked properly, you can do the following:

cd L1Trigger/L1TMuonEndCap/test/tools/
root -l pc_luts/firmware_data/validate.root
## EMTF emulator vs. CMS simulation phi or theta coordinate
tree->Draw("fph_emu - fph_sim : fph_sim >> dPh_vs_phi(360,-180,180,80,-0.5,0.5)","","colz")
tree->Draw("fth_emu - fth_sim : fph_sim >> dTh_vs_phi(360,-180,180,80,-1.0,1.0)","","colz")
## Look at a specific region, e.g. ME+1/1a
tree->Draw("fph_emu - fph_sim : fph_sim >> dPh_vs_phi(360,-180,180,80,-0.5,0.5)","(endcap == 1 && station == 1 && ring == 4)","colz")

