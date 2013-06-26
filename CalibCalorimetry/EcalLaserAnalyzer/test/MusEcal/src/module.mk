# Source files to compile 
FILES := MERun MERunManager
FILES += METimeInterval MEVarVector
FILES += MusEcal
FILES += MECanvasHolder
FILES += MEEBDisplay MEEEDisplay
FILES += MusEcalGUI 
FILES += MEClickableCanvas MEPlotWindow 
FILES += MERunPanel MEChanPanel MELeafPanel MEMultiVarPanel 

# Header files to use for dictionary generation
DICTFILES := $(FILES) LinkDef

# Executable files
PROGRAMS := createEBHist
PROGRAMS += getChannel
PROGRAMS += runTest
PROGRAMS += runGUI
PROGRAMS += runGeom
PROGRAMS += writePrim

