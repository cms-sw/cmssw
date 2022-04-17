include ../../Makefile.config

all: libConfigDict.so

libConfigDict.so: ConfigDict.cc
	${CXX} -I.. -I../.. -I$(shell root-config --incdir) ${CPPFLAGS} -fPIC -shared -o $@ $^

ConfigDict.cc: ../IterationConfig.h ../SteeringParams.h ConfigLinkDef.h
	rootcling -f -I=.. -I=../.. $@ $^

clean:
	rm -f libConfigDict.so ConfigDict.cc ConfigDict_rdict.pcm
	find . -size 0 -delete # rootcling leaves some junk around
