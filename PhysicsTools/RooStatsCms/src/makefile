# NOTE: this is now fully automatic. You are not supposed to modify it,
# providing that you use the usual file naming rules.
LIB	      = ../lib
INCL      = ../interface
TESTDIR	  = ../test
BIN	      = ../bin
OBJ	      = ../obj
ObjSuf        = o
SrcSuf        = cxx
DllSuf        = so
OutPutOpt     = -o

ROOTCFLAGS    = $(shell root-config --cflags)
ROOTLIBS      = $(shell root-config --libs --glibs)
ROOTGLIBS     = $(shell root-config --glibs)
ROOTLDFLAGS   = $(shell root-config --ldflags)


CPPFLAGS      = -I${INCL}/
CXX 		= /usr/bin/g++ -g  -D STANDALONE -Wall
CXXFLAGS      = -O -fPIC $(ROOTCFLAGS) 
LD			= /usr/bin/g++
LDFLAGS       = $(ROOTLDFLAGS) -rdynamic -D STANDALONE

OWN_LIB		= -l RooStatsCms
SONAME	      = libRooStatsCms.so
SOFLAGS       = -shared -Wl,-soname,$(SONAME)
LIBS          = $(ROOTLIBS) -l RooFit -l RooFitCore -l Minuit
#GLIBS         = $(ROOTGLIBS)  -L/usr/X11R6/lib64 -lXpm -lX11 -lm -ldl -rdynamic
GLIBS         = $(ROOTGLIBS)

HDRS          = $(patsubst cintdict.h,,$(wildcard ${INCL}/*.h))

SRCS          = $(patsubst cintdict.cc,,$(wildcard *.cc) $(wildcard *.C))
TEST          = $(wildcard ../bin/*.cc) 
TEST_HDRS     = $(wildcard ..//*.h)

OBJ_TMP       = $(patsubst main.o,,$(SRCS:.cc=.o)  cintdict.o)
#OBJS     = $(OBJ_TMP:.C=.o)
OBJS_TMP2     = $(OBJ_TMP:.C=.o)
OBJS          = $(addprefix ${OBJ}/,${OBJS_TMP2})

ICCS          = $(wildcard *.icc)

BINS	      = $(wildcard ../test/*.cpp)
PROGRAM_TMP       = $(BINS:.cpp=.exe)
PROGRAM_TMP2       = $(notdir $(PROGRAM_TMP))
PROGRAM		= $(addprefix ../bin/,${PROGRAM_TMP2})

#all:            $(PROGRAM)
all:		${LIB}/$(SONAME)

progs:
		@echo   "LinkDef file: $(LINKDEF)"
		@echo   "Header files: $(HDRS)"
		@echo   "Source files: $(SRCS)"
		@echo 	"Object files: $(OBJS)"
		@echo 	"Dict. Hdrs:   $(DICTHDRS)"	
		@echo 	"Program:   $(PROGRAM)"

exe: 	$(PROGRAM)

${LIB}/$(SONAME):	$(OBJS)
		@echo "Linking $(SONAME):"
		$(LD) $(LDFLAGS) $(OBJS) $(SOFLAGS) -o ${LIB}/$(SONAME) ${LIBS}
		#rm cintdict*

clean:          cleandict
		rm -f $(OBJS) main.o core

cleandict:	
		rm -f cintdict.*


# executables
# ${BIN}/%.exe : ${TESTDIR}/%.cpp $(TEST) $(TEST_HDRS) $(HDRS) ${LIB}/$(SONAME)
# 		$(CXX) $< -o $@  $(TEST) -L${LIB} -l RooStatsCms  $(CPPFLAGS) ${GLIBS} ${LIBS} ${CXXFLAGS}  
${BIN}/%.exe : ${TESTDIR}/%.cpp $(TEST) $(TEST_HDRS) $(HDRS) ${LIB}/$(SONAME)
		$(CXX) $< -o $@  $(TEST) $(CPPFLAGS) ${CXXFLAGS} -L${LIB} $(OWN_LIB) ${LIBS} 
# Generic dependencies (add dependency on .h)
${OBJ}/%.o : %.cc ${INCL}/%.h
	$(CXX) -c  $(CXXFLAGS) $(CPPFLAGS) $< -o $@

${OBJ}/cintdict.o : cintdict.cc cintdict.h
	$(CXX) -c  $(CXXFLAGS) $(CPPFLAGS) $< -o $@

${OBJ}/%.o : %.C %.h
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $< -o $@


LINKDEF		= $(wildcard ${INCL}/*LinkDef.h)
#Put the LinkDef.h at the end of the list.
DICTHDRS        = $(patsubst $(LINKDEF),,$(HDRS)) $(LINKDEF)

#DICTHDRS = "interface/ConstrBlock3.h interface/LikelihoodCalculator.h interface/PdfCombiner.h"
#       interface/RatioFinderResults.h  interface/RscConstrArrayFiller.h  interface/StatisticalPlot.h
# interface/binomial_intervals.h             interface/ConstrBlockArray.h   interface/LimitCalculator.h       interface/PLScan.h           interface/RscAbsPdfBuilder.h    interface/Rsc.h
# interface/binomial_noncentral_interval.h   interface/ExclusionBandPlot.h  interface/LimitPlot.h             interface/PLScanPlot.h       interface/RscBaseModel.h        interface/RscMultiModel.h
# interface/binomial_noncentral_intervals.h  interface/FCCalculator.h       interface/LimitResults.h          interface/PLScanResults.h    interface/RscCatModel.h         interface/RscTool.h
# interface/Constraint.h                     interface/FCResults.h          interface/Minus2LnQCalculator.h   interface/RatioFinder.h      interface/RscCombinedModel.h    interface/RscTotModel.h
# interface/ConstrBlock2.h                   interface/LEPBandPlot.h        interface/NLLPenalty.h            interface/RatioFinderPlot.h  interface/RscCompModel.h        interface/StatisticalMethod.h


cintdict.cc: $(DICTHDRS)
	    @echo "Generating dictionary for  ..."
	    rootcint -f cintdict.cc -c -p -I${ROOTSYS}/interface ${CPPFLAGS} $(DICTHDRS)
