# -*-make-*-
#  Autors:   Dmytro Kovalskyi
#            Johannes Muelmenstaedt
# 
ifndef VERBOSE
  QUIET := @
endif

all: core display

core:   env Makefile.core
	$(QUIET) $(MAKE) -f Makefile.core
	
display: env Makefile.project
	$(QUIET) $(MAKE) -f Makefile.project

# check if proper environment is setup
env: 
	$(QUIET) if [ -z "$(ROOTSYS)" ]; then \
	   echo "ROOTSYS not set. Cannot compile dictionaries."; \
	   exit 1; \
	fi;

clean: cleancore cleandisplay

cleancore: Makefile.core
	$(QUIET) $(MAKE) -f Makefile.core clean
	
cleandisplay: Makefile.project
	$(QUIET) $(MAKE) -f Makefile.project clean
