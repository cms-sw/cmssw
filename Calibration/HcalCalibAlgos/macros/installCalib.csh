#!/bin/csh

g++ -Wall -Wno-deprecated -I./ `root-config --cflags` CalibMain.C -o CalibMain.exe `root-config --libs`
