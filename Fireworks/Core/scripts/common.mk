######################################################################################
# 
#                General rules for making object files
# 
######################################################################################
# get object files
tmp/%.o:   %.cc
	$(QUIET) echo "compiling $<"; \
	mkdir -p $(dir $@); \
	$(CC) $(CFLAGS) $< -c -o $@

# get object files for C code
tmp/%.co:   %.c
	$(QUIET) echo "compiling $<"; \
	mkdir -p $(dir $@); \
	$(CC) $(CFLAGS) $< -c -o $@

tmp/%.cpp:  %.h %_def.xml
	$(QUIET) echo "generating dictionaries based on $*_def.xml"; \
	mkdir -p $(dir $@); \
	$(FWROOTSYS)/bin/genreflex $*.h -s $*_def.xml -o $@ $(INCLUDE) --gccxmlpath=external/gccxml/bin --gccxmlopt="--gccxml-compiler $(CC)" 
# dictionary creation

# dictionary creation
# NOTE: for some reason I needed to add the original
#   LinkDef file contect to the generated dictionary code to be 
#   able to compile
tmp/%LinkDef.cc:  %LinkDef.h
	$(QUIET) echo "generating ROOT dictionaries based on $<"; \
	mkdir -p $(dir $@); \
	LD_LIBRARY_PATH=$(FWROOTSYS)/lib; export LD_LIBRARY_PATH; \
	DYLD_LIBRARY_PATH=$(FWROOTSYS)/lib; export DYLD_LIBRARY_PATH; \
	ROOTSYS=$(FWROOTSYS); export ROOTSYS; \
	$(FWROOTSYS)/bin/rootcint -f $@.tmp -c -p $(INCLUDE) $<; \
	cat $< $@.tmp > $@

# object files for dictionaries
%.do:   %.cpp
	$(QUIET) echo "compiling dictionaries $<"; \
	$(CC) $(CFLAGS) -I$(FWROOTSYS)/include $< -c -o $@

# object files for dictionaries
%.ro:   %.cc
	$(QUIET) echo "compiling ROOT dictionaries $<"; \
	$(CC) $(CFLAGS) $(INCLUDE) $< -c -o $@

# check dependencies
# -lRIO -lNet - "hacks" for standard ROOT to be able to use the
# recipe below to check for undefined symbols
tmp/%.out:  %
	$(QUIET) echo "checking shared library for missing symbols: $<"; \
	echo "int main(){}" > tmp/$<.cpp; \
	$(CC) $(CFLAGS) -Wl,-rpath -Wl,./ -L$(FWROOTSYS)/lib $(addprefix $(LinkerOptions),$(LDLIBRARYPATH)) -lRIO -lNet -o $@ $< tmp/$<.cpp

# this recipe for dependency generation is the one recommended by the
# gmake manual, with slight tweaks for the location of the output
tmp/%.d:   %.cc
	$(QUIET) echo "dependencies for $<"; \
	mkdir -p $(dir $@); \
	$(CC) -M -MT $@ -MT ${@:.d=.o} $(CFLAGS) $< > $@; \
                     [ -s $@ ] || rm -f $@
-include /dev/null $(ProjectObjects:.o=.d)
-include /dev/null $(CoreObjects:.o=.d)

tmp/%LinkDef.d:  %LinkDef.h
	$(QUIET) echo "dependencies for ROOT dictionaries based on $<"; \
	mkdir -p $(dir $@); \
	$(CXX) -M -MT $@ -MT ${@:.d=.ro} -MT ${@:.d=.cc} $(CFLAGS) $(INCLUDE) $< > $@; \
                     [ -s $@ ] || rm -f $@
-include /dev/null $(ProjectRootDicSources:.cc=.d)
-include /dev/null $(CoreRootDicSources:.cc=.d)

tmp/%classes.d : %classes.h
	$(QUIET) echo "dependencies for dictionaries based on $*classes_def.xml"; \
	mkdir -p $(dir $@); \
	$(CXX) -M -MT $@ -MT ${@:.d=.do} -MT ${@:.d=.cpp} $(CFLAGS) $(INCLUDE) $< > $@; \
                     [ -s $@ ] || rm -f $@

-include /dev/null $(ProjectDictionarySources:.cpp=.d)
-include /dev/null $(CoreDictionarySources:.cpp=.d)
