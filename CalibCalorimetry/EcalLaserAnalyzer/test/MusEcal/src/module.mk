# Source files to compile 
FILES := MERun MERunManager
FILES += METimeInterval MEVarVector
FILES += MusEcal
FILES += MECanvasHolder
FILES += MEEBDisplay MEEEDisplay
FILES += MusEcalGUI 
FILES += MEClickableCanvas MEPlotWindow 
FILES += MERunPanel MEChanPanel MELeafPanel MEMultiVarPanel 
FILES += METimeInterval MEIntervals
FILES += MECorrector2Var METwoVarPanel

# Header files to use for dictionary generation
DICTFILES := $(FILES) LinkDef

# Executable files
PROGRAMS := createEBHist
PROGRAMS += getChannel
PROGRAMS += runPatrick
PROGRAMS += runMatthieu
#PROGRAMS += runTest2
PROGRAMS += runStab
PROGRAMS += runTest
PROGRAMS += runGUI
#PROGRAMS += runGeom
PROGRAMS += writePrim
#PROGRAMS += plot

