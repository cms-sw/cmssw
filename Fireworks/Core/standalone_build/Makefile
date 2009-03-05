# -*-make-*-
#  Autors:   Dmytro Kovalskyi
#            Johannes Muelmenstaedt
# 
ifndef VERBOSE
  QUIET := @
endif

ifeq ($(shell uname), Linux)
	SO_EXT := so
else 
	ifeq ($(shell uname), Darwin)
		SO_EXT := dylib
	endif
endif

export FWROOTSYS = external/root

all: display install

core:   Makefile.core
	$(QUIET) $(MAKE) -f Makefile.core

display: Makefile.project
	$(QUIET) $(MAKE) -f Makefile.project

install: Makefile.core
	$(QUIET) $(MAKE) -f Makefile.core install;\
	echo Done

clean: cleandisplay

cleanall: cleancore cleandisplay

cleancore: 
	$(QUIET) echo "removing temporary core objects and output library..."; \
	rm -rf tmp/cms; rm libCMSDataFormats.$(SO_EXT); echo "done"

cleandisplay: 
	$(QUIET) echo "removing temporary project objects and output libraries..."; \
	rm -rf tmp/src; rm libCMSShow.$(SO_EXT); echo "done"

