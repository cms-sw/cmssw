# Simple makefile

EXE=bin/createPerfNumDenHists

SOURCES=$(wildcard src/*.cc)
OBJECTS=$(SOURCES:.cc=.o)
HEADERS=$(SOURCES:.cc=.h)

CC          = g++
CXX         = g++
CXXFLAGS    = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual
LD          = g++
LDFLAGS     = -g -O2
SOFLAGS     = -g -shared
CXXFLAGS    = -std=c++20 -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual
LDFLAGS     = -g -O2
ROOTLIBS    = $(shell root-config --libs)
ROOTCFLAGS  = $(shell root-config --cflags)
CXXFLAGS   += $(ROOTCFLAGS)
CFLAGS      = $(ROOTCFLAGS) -Wall -Wno-unused-function -g -O2 -fPIC -fno-var-tracking
EXTRACFLAGS = $(shell rooutil-config) -I../code/core
EXTRAFLAGS  = -fPIC -ITMultiDrawTreePlayer -Wunused-variable -lTMVA -lEG -lGenVector -lXMLIO -lMLP -lTreePlayer $(shell rooutil-config)

all: rooutil $(EXE)

$(EXE): src/performance.o src/helper.o src/LSTEff.o
	$(LD) $(CXXFLAGS) $(LDFLAGS) src/performance.o src/helper.o src/LSTEff.o $(ROOTLIBS) $(EXTRAFLAGS) -o $@

src/LSTEff.o: src/LSTEff.cc src/LSTEff.h
	$(CC) $(CFLAGS) $(EXTRACFLAGS) $< -c -o $@

src/performance.o: src/performance.cc src/performance.h
	$(CC) $(CFLAGS) $(EXTRACFLAGS) $< -c -o $@

src/helper.o: src/helper.cc src/helper.h
	$(CC) $(CFLAGS) $(EXTRACFLAGS) $< -c -o $@

rooutil:
	$(MAKE) -C ../code/rooutil

clean:
	rm -r src/performance.o src/helper.o src/LSTEff.o $(EXE)

.PHONY: all clean rooutil
