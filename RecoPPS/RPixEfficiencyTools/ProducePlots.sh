#!/bin/bash
if [ $# -ne 1 ]
then
	echo "Run Number required. Nothing done."
else
	root -l -q -b "ViewPlots.C($1,true)"
	echo ""
fi