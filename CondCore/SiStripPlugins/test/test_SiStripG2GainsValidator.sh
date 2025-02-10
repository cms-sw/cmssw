#!/bin/bash

function die { echo $1: status $2 ; exit $2; }
SiStripG2GainsValidator --user-mode || die "Failure running SiStripG2GainsValidator" $?
