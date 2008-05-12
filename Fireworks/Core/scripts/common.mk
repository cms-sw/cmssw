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

# dictionary creation
tmp/%.cpp:  %.h %_def.xml
	$(QUIET) echo "generating dictionaries based on $*_def.xml"; \
	mkdir -p $(dir $@); \
	$(ROOTSYS)/bin/genreflex $*.h -s $*_def.xml -o $@ $(INCLUDE) --gccxmlpath=external/gccxml/bin

# dictionary creation
# NOTE: for some reason I needed to add the original
#   LinkDef file contect to the generated dictionary code to be 
#   able to compile
tmp/%LinkDef.cc:  %LinkDef.h
	$(QUIET) echo "generating ROOT dictionaries based on $<"; \
	mkdir -p $(dir $@); \
	LD_LIBRARY_PATH=$(ROOTSYS)/lib; export LD_LIBRARY_PATH; \
	$(ROOTSYS)/bin/rootcint -f $@.tmp -c -p $(INCLUDE) $<; \
	cat $< $@.tmp > $@

# object files for dictionaries
%.do:   %.cpp
	$(QUIET) echo "compiling dictionaries $<"; \
	$(CC) $(CFLAGS) -I$(ROOTSYS)/include $< -c -o $@

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
	$(CC) $(CFLAGS) -Wl,-rpath -Wl,./ -L$(ROOTSYS)/lib $(addprefix $(LinkerOptions),$(LDLIBRARYPATH)) -lRIO -lNet -o $@ $< tmp/$<.cpp

