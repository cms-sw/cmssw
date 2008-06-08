# -*-make-*-
#  Autors:   Dmytro Kovalskyi
#            Johannes Muelmenstaedt
# 
ifndef VERBOSE
  QUIET := @
endif

ROOTSYS = external/root

all: display install

core:   Makefile.core
	$(QUIET) $(MAKE) -f Makefile.core
	
display: Makefile.project
	$(QUIET) $(MAKE) -f Makefile.project

install: Makefile.core
	$(QUIET) $(MAKE) -f Makefile.core install;\
	echo Done

# check if proper environment is setup
#env: 
#	$(QUIET) if [ -z "$(ROOTSYS)" ]; then \
#	   echo "ROOTSYS not set. Abort."; \
#	   exit 1; \
#	fi;

clean: cleancore cleandisplay

cleancore: Makefile.core
	$(QUIET) $(MAKE) -f Makefile.core clean
	
cleandisplay: Makefile.project
	$(QUIET) $(MAKE) -f Makefile.project clean
