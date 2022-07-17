#!/bin/bash

verbose=false
testing=false

prefix=/nfshome0/dqmpro/BeamMonitorDQM

sourceFile=$prefix/BeamFitResultsForDIP.txt
sourceFile1=$prefix/BeamFitResultsOld_TkStatus.txt

beamSpotDipStandalone $verbose $testing $sourceFile $sourceFile1

