#------------------------------------------------------------------------------

CXX      = c++
CFLAGS   = -O3 -Wall

#------------------------------------------------------------------------------

all:		parser

parser:		parser.cc
		$(CXX) $(CFLAGS) -o parser parser.cc $(LIBS) 

clean:
		@rm -f *~ *.o parser

