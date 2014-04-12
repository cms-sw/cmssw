FFLAGS   = -O0 -ffixed-line-length-none -g

FC       = f77
LIBS     = -lm 
INCLUDES = -I./
SRCS     = $(wildcard *.f)
OBJS     = $(patsubst %.f,%.o,$(SRCS))
HDRSI    = $(wildcard *.inc) 
HDRSH    = $(wildcard *.h) 

all: hardcol

hardcol: ${OBJS} hardcol.o
	@echo "Mode 1" 
	${FC} ${FFLAGS} ${INCLUDES} -o $@ ${OBJS} hardcol.o ${LIBS}
	@echo " "


.f.o:
	@echo "1 Working on  $<  |  $@ "
	${FC} ${FFLAGS} ${INCLUDES} -c -o $@ $<
	@echo " "

.F.o:
	@echo "2 Working on  $<  |  $@ "
	${FC} ${FFLAGS} ${INCLUDES} -c -o $@ $<
	@echo " "

depend: 
	makedepend ${SRCS}

clean:
	rm -rf *.o core *~ hardcol

tar:
	tar cf code.tar Makefile *.F *.f *.inc *.h 

print:
	more Makefile $(HDRSI) $(HDRSH) $(SRCS) | enscript -2r -p listing.ps

