all : ../bin/pixelAnalyzer

../bin/pixelAnalyzer : pixelAnalyzer.cc
	g++ -Wall -O3 `root-config --cflags` -o $@ $< `root-config --libs` -lm

