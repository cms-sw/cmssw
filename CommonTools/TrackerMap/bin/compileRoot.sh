#!/bin/sh

g++ -g -m32 -Wall -ansi -I $ROOTSYS/include -L $ROOTSYS/lib `root-config --glibs` -o svgTopng.exe svgTopng.C

