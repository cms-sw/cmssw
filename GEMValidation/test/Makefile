# Root configuration                                                                                                                                                      
ROOTCFLAGS = $(shell root-config --cflags)
ROOTLIBS   = $(shell root-config --libs --glibs)

# set the compiler options                                                                                                                                                
CXX       = g++                 #use g++ as compiler                                                                                                                      
CXXFLAGS  = -g                  #compile with debug information                                                                                                           
CXXFLAGS += -O2                 #optimize                                                                                                                                 
CXXFLAGS += -Wall               #show warnings for everything                                                                                                             
CXXFLAGS += -fPIC               #compile source file as sharable object                                                                                                   
CXXFLAGS += -c                  #do not link, produce only .o files                                                                                                       
CXXFLAGS += $(ROOTCFLAGS)

# set the linker options                                                                                                                                                  
LD            = g++                     #use g++ for linking                                                                                                              
LDFLAGS       = -O                      #-O optimize                                                                                                                      

LIBS          = $(ROOTLIBS)

PROJECT = runGEMValidator
OBJ = $(PROJECT).o  SimHitValidator.o DigiValidator.o #BaseValidator.o

# compile all programs                                                                                                                                                   
all: $(PROJECT)

$(PROJECT): $(OBJ)
	$(LD) $(LDFLAGS) $(OBJ) $(LIBS) -o $(PROJECT)

%.o: %.cc %.h
	$(CXX) $(CXXFLAGS) -c $<

PHONY : clean

clean:
	@rm -rf *.o *~
