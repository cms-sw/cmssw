FFLAGS   = -O0 -ffixed-line-length-none -g

FC       = gfortran
LIBS     = -lm 
INCLUDES = -I./
SRCS     = $(wildcard *.F)
OBJS     = $(patsubst %.F,%.o,$(SRCS))
HDRSI    = $(wildcard *.inc) 
HDRSH    = $(wildcard *.h) 

all: bcvegpy

bcvegpy: ${OBJS} bcvegpy.o
	@echo "Mode 1" 
	${FC} ${FFLAGS} ${INCLUDES} -o $@ ${OBJS} bcvegpy.o ${LIBS}
	@echo " "


.F.o:
	@echo "1 Working on  $<  |  $@ "
	${FC} ${FFLAGS} ${INCLUDES} -c -o $@ $<
	@echo " "

.f.o:
	@echo "2 Working on  $<  |  $@ "
	${FC} ${FFLAGS} ${INCLUDES} -c -o $@ $<
	@echo " "

depend: 
	makedepend ${SRCS}

clean:
	rm -rf *.o core *~ bcvegpy

tar:
	tar cf code.tar Makefile *.F *.f *.inc *.h 

print:
	more Makefile $(HDRSI) $(HDRSH) $(SRCS) | enscript -2r -p listing.ps


