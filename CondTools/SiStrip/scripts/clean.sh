#!/bin/sh

rm -f RunToBeSubmitted RunList AddedRuns
rm -f lastRun out.txt physicsRuns.txt latencyRuns.txt
rm -f RunToDoO2O* RunToBeSubmitted*
rm -f lockFile
echo 0 > lastRun
