SRC=src


# command for documentation generation
DOXYGEN=doxygen


.PHONY: all clean cleandict exe doc

all:
	mkdir -p lib	
	mkdir -p obj
	mkdir -p bin
	$(MAKE) -C $(SRC)

clean:
	$(MAKE) -C $(SRC) clean
	rm -rf lib/*
	rm -rf lib
	rm -rf obj/*
	rm -rf obj
	rm -rf bin/*.exe
exe: 
	$(MAKE) -C $(SRC) exe

progs:
	$(MAKE) -C $(SRC) progs

doc:
	$(DOXYGEN) scripts/Doxygen.cfg

