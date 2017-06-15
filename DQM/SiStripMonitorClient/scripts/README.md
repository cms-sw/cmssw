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
Please uncomment the line 166 to line 167 in TkMap_script_phase1.py before you start running the script. It will add color lengend on QT TkMap. This script only works on vocms061