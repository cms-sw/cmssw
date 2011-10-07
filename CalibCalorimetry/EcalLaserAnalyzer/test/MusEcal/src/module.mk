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
FILES += MENormManager MENLS
FILES += MECLSManager 
# Header files to use for dictionary generation
DICTFILES := $(FILES) LinkDef

# Executable files
PROGRAMS := createEBHist
PROGRAMS += getChannel
#PROGRAMS += runStab
PROGRAMS += runStabBis
PROGRAMS += runStabTP
PROGRAMS += runTest
#PROGRAMS += runExample
PROGRAMS += runGUI
PROGRAMS += plotBis
PROGRAMS += plotTP
PROGRAMS += writePrim
PROGRAMS += testPrim
PROGRAMS += writePrimEE
PROGRAMS += writePrimEBP
PROGRAMS += writeNLS
