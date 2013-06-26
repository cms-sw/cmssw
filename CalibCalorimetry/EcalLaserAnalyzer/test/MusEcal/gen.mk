
# module.mk appends to FILES and DICTFILES
FILES :=
DICTFILES :=

# including the module.mk file
-include $(patsubst %, %/module.mk,src)

# appending the values found in FILES to the variable SRC
SRC += $(patsubst %,src/%,$(FILES))

# appending the values found in DICTFILES to DICTH_modulename
DICTH_src := $(foreach i, $(patsubst %,src/%,$(DICTFILES)), $(wildcard $(i).h) $(wildcard $(i).hh))
# if dict header files exist, append to variable SRC
ifneq ($(DICTH_src),)
SRC += src/$(PKGNAME)_dict_src
endif

PROG += $(patsubst %,$(BINDIR)/%,$(PROGRAMS))

# appending the values found in FILES to the variable SRC
PROGSRC += $(patsubst %,src/%,$(PROGRAMS))

# VPATH += :src

# make sure the output directories are there
__dummy := $(shell mkdir -p $(DEPDIR)/src $(OBJDIR)/src)

# a couple of rules to copy executable files correctly
$(BINDIR)/%: src/%
	cp $^ $@

#$(BINDIR)/%: src/%.bin
#	cp $^ $@

$(BINDIR)/%: src/bin/%
	cp $^ $@

$(BINDIR)/%: ${OBJDIR}/src/%.bin
	cp $^ $@

