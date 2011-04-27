################################################################################
# HiggsAnalysis/Combined Limit Makefile                                        #
#                                                                              #
# $Author$ - $Date$                                                            #
#                                                                              #
# o Automatic compilation of new programs and classes*.                        #
# o Automatic generation of CINT dictionaries via rootcint.                    #
#                                                                              #
# * progs should have cpp extension, classes .cc and headers .h                # 
#                                                                              #
################################################################################


# Boost
BOOST_INC = 

# Compiler and flags -----------------------------------------------------------
CC = g++

ROOTCFLAGS = $(shell root-config --cflags)
ROOTLIBS = $(shell root-config --libs --glibs)
ROOTINC = $(shell root-config --incdir)

CCFLAGS = -D STANDALONE $(ROOTCFLAGS) $(BOOST_INC) -O -Wall -g -fPIC
LIBS = $(ROOTLIBS) -l RooFit -lRooFitCore -l RooStats -l Minuit -l Foam -lboost_filesystem -lboost_program_options

# Library name -----------------------------------------------------------------
LIBNAME=CombinedLimit
SONAME=lib$(LIBNAME).so

# Linker and flags -------------------------------------------------------------
LD = g++
ROOTLDFLAGS   = $(shell root-config --ldflags)
LDFLAGS       = $(ROOTLDFLAGS) -rdynamic -shared -Wl,-soname,$(SONAME) -fPIC

# Dictionaries filename --------------------------------------------------------
DICTNAME=cintdictionary

# Directory structure ----------------------------------------------------------
SRC_DIR = src
INC_DIR = interface
LIB_DIR = lib
PROG_DIR = bin
EXE_DIR = exe
OBJ_DIR = obj


# Useful shortcuts -------------------------------------------------------------
SRCS = $(notdir $(wildcard $(SRC_DIR)/*.cc))
SRCS += $(DICTNAME).cc
OBJS = $(SRCS:.cc=.o)
PROGS = $(notdir $(wildcard ${PROG_DIR}/*.cpp)) 
EXES = $(PROGS:.cpp=)

# Classes with dicts -----------------------------------------------------------
DICTHDRS= $(notdir $(shell grep -l ClassDef interface/*h))

#Makefile Rules ---------------------------------------------------------------
.PHONY: clean dirs dicr obj lib exe

all: dirs dict obj lib exe

#---------------------------------------

dirs:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(EXE_DIR)

#---------------------------------------

dict: dirs $(SRC_DIR)/$(DICTNAME).cc
$(SRC_DIR)/$(DICTNAME).cc $(INC_DIR)/$(DICTNAME).h :
# 	@echo "\n*** Generating dictionaries ..."
	rootcint -f $(SRC_DIR)/$(DICTNAME).cc -c -I$(INC_DIR) -I$(ROOTINC) $(DICTHDRS) $(SRC_DIR)/LinkDef.h

#---------------------------------------

obj: dict $(addprefix ${SRC_DIR}/,${SRCS})
# 	@echo "\n*** Compiling ..."
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cc
	$(CC) $(CCFLAGS) -I $(INC_DIR) -c $< -o $@

#---------------------------------------

lib: dirs ${LIB_DIR}/$(SONAME)
${LIB_DIR}/$(SONAME):$(addprefix $(OBJ_DIR)/,$(OBJS))
# 		@echo "\n*** Building $(SONAME) library:"
		$(LD) $(LDFLAGS) $(BOOST_INC)  -fPIC $(addprefix $(OBJ_DIR)/,$(OBJS))  $(SOFLAGS) -o $@ $(LIBS)

#---------------------------------------

exe: $(addprefix $(EXE_DIR)/,$(EXES))
# 	@echo "\n*** Compiling executables ..."
$(addprefix $(EXE_DIR)/,$(EXES)) : $(addprefix $(PROG_DIR)/,$(PROGS))
	$(CC) $< -o $@ $(CCFLAGS) -L $(LIB_DIR) -l CombinedLimit -I $(INC_DIR) $(BOOST_INC) $(LIBS)


#---------------------------------------

clean:
# 	@echo "*** Cleaning all directories and dictionaries ..."
	@rm -rf $(OBJ_DIR) 
	@rm -rf $(EXE_DIR)
	@rm -rf $(LIB_DIR)
	@rm -rf $(SRC_DIR)/$(DICTNAME)*

#---------------------------------------
