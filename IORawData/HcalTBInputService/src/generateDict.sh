#!/bin/sh

rm -f CDFChunk_dict.*
rootcint -f CDFChunk_dict.cc -c CDFChunk.h CDFEventInfo.h
