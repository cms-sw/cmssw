Phase1 PixelMaps Scripts
==================

The script *TkMap_script_phase1.py* is used to replace the old script *TkMap_script_automatic_DB.sh*. 

New Features:
Ability to produce Pixel Phase I maps and input trees for TkCommissioner using code developed by Pawel
Shifters have the option to continue or stop the script when the DQM file is not fully processed



How to use
----------

`./TkMap_script_phase1.py Dataset RunNumber1 RunNumber2 ... `

IMPORNTANT
----------
Please uncomment the color legend section before you start running the script. It will add color lengend on QT TkMap. This script only works on vocms061

Please comment and uncomment the global tag section after the production switch to 9_3_X release.

PixelMapPlotter(DeadROCViewer)
=============

See full documentation here:

https://twiki.cern.ch/twiki/bin/view/CMS/DeadROCViewer


TH2PolyOfflineMaps
==================

The script which behaviour is very similar to the https://github.com/pjurgielewicz/cmssw/tree/pjAnalyzerBranch/DQM/SiPixelPhase1Analyzer in the *MODE_REMAP* but it additionally produces full Tracker Maps (Barrel + Pixel) in the single image for all module level plots available in the input file.

Moreover it looks for 20 minimum and maximum values in Tracker Map bins and prints them (with a corresponding det ID) in the output text file.

How to use
----------

`python TH2PolyOfflineMaps.py <name of the input file> <width (px)> <height (px)> <limits file name>`

where the run number has to be able to be deducted from the input file name. Supported format is as follows

`*_R000######*` - run number is a 6-digit value.

Limits file is an optional file which can help to set custom and fixed z-axis range for a given map. Description of the map range consists of exactly 4 elements:
  1. Map name
  2. Minimum z value
  3. Maximum z value
  4. Set log axis (anything different than '0' is considered as True)
  
You can specify limits for different maps in different rows in this file. Empty lines and lines starting with '#' character are skipped during parsing stage. If you do not specify limits for a map which is created by the script it's range will be adjusted automatically.

Outputs (maps + text file) are saved inside `.OUT/`.

IMPORTANT
---------

REALTIVE POSITIONING IS MESSY IN THE CURRENT VERION OF PYROOT. IT CAN CHANGE FROM VERSION TO VERSION, SO YOU HAVE TO ADJUST IT FOR YOUR NEEDS.

ALSO LATEX FUNCTIONALITY IS NOT WORKING.

DeadROC_duringRun
=================
Count and compare the dead roc at beginning and the end of the run

PhaseITreeProducer
==================

Produce the tree for TkCommissioner