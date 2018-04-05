.PHONY: all clean

all: uGTrates

clean:
	rm -f uGTrates

uGTrates: uGTrates.cc
	g++ uGTrates.cc -o uGTrates $(shell root-config --cflags --ldflags --libs)
