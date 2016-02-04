# temporarily in CVS for DQM needs

CXX     = g++

CXXFLAGS  = -pedantic -Wno-long-long -Wall 

DEFINES   = $(INPDEF) $(OUTDEF)

#------------------------------------------------------
INCLUDES += -I../../ -I../../../ -I../../../../../ 


SRCPATH   = ../src
SRCS      = $(SRCPATH)/CSCALCTDigi.cc $(SRCPATH)/CSCCLCTDigi.cc\
            $(SRCPATH)/CSCWireDigi.cc $(SRCPATH)/CSCComparatorDigi.cc \
	    $(SRCPATH)/CSCStripDigi.cc $(SRCPATH)/CSCRPCDigi.cc \
	    $(SRCPATH)/CSCCorrelatedLCTDigi.cc

OBJS      = $(notdir $(SRCS:.cc=.o))

%.o: ../src/%.cc ../interface/%.h
	$(CXX) -c $(CXXFLAGS) $(DEFINES) $(INCLUDES) $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(DEFINES) $(INCLUDES) $<


all: lib 

lib: libCSCDigi.a libCSCDigi.so


libCSCDigi.so: $(OBJS)
	$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) -shared $(OBJS) $(TriDASLIB) -o $@

libCSCDigi.a: $(OBJS)
	ar cr $@ $(OBJS)

clean:
	@rm -f libCSCDigi.* *.o program RunNum* core*
