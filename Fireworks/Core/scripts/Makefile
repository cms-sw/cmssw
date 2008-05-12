# -*-make-*-
#  Autors:   Dmytro Kovalskyi
#            Johannes Muelmenstaedt
# 
ifndef VERBOSE
  QUIET := @
endif

all: display install

core:   env Makefile.core
	$(QUIET) $(MAKE) -f Makefile.core
	
display: env Makefile.project
	$(QUIET) $(MAKE) -f Makefile.project

install: env Makefile.core
	$(QUIET) $(MAKE) -f Makefile.core install;\
	echo Done

# check if proper environment is setup
env: 
	$(QUIET) if [ -z "$(ROOTSYS)" ]; then \
	   echo "ROOTSYS not set. Abort."; \
	   exit 1; \
	fi;

clean: cleancore cleandisplay

cleancore: Makefile.core
	$(QUIET) $(MAKE) -f Makefile.core clean
	
cleandisplay: Makefile.project
	$(QUIET) $(MAKE) -f Makefile.project clean
