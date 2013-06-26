#!/bin/bash


modules=$*

function makegen
{
    mod=$1
    cat >> gen.mk <<EOF

# module.mk appends to FILES and DICTFILES
FILES :=
DICTFILES :=

# including the module.mk file
-include \$(patsubst %, %/module.mk,${mod})

# appending the values found in FILES to the variable SRC
SRC += \$(patsubst %,${mod}/%,\$(FILES))

# appending the values found in DICTFILES to DICTH_modulename
DICTH_${mod} := \$(foreach i, \$(patsubst %,${mod}/%,\$(DICTFILES)), \$(wildcard \$(i).h) \$(wildcard \$(i).hh))
# if dict header files exist, append to variable SRC
ifneq (\$(DICTH_${mod}),)
SRC += ${mod}/\$(PKGNAME)_dict_${mod}
endif

PROG += \$(patsubst %,\$(BINDIR)/%,\$(PROGRAMS))

# appending the values found in FILES to the variable SRC
PROGSRC += \$(patsubst %,${mod}/%,\$(PROGRAMS))

# VPATH += :${mod}

# make sure the output directories are there
__dummy := \$(shell mkdir -p \$(DEPDIR)/${mod} \$(OBJDIR)/${mod})

# a couple of rules to copy executable files correctly
\$(BINDIR)/%: ${mod}/%
	cp \$^ \$@

#\$(BINDIR)/%: ${mod}/%.bin
#	cp \$^ \$@

\$(BINDIR)/%: ${mod}/bin/%
	cp \$^ \$@

\$(BINDIR)/%: \${OBJDIR}/${mod}/%.bin
	cp \$^ \$@

EOF
}

function makedictgen
{
    mod=$1
    cat >> dictgen.mk <<EOF
${mod}/\$(PKGNAME)_dict_${mod}.cc: \$(DICTH_${mod})
	@echo "Generating dictionary \$@..."
	@\$(ROOTCINT) -f \$@ -c \$(filter -I% -D%,\$(CXXFLAGS)) \$^
EOF
}


rm -f gen.mk dictgen.mk

for i in $modules; do
    makegen $i
    makedictgen $i
done

