#!/bin/bash

cd ~/$CMSSW_VERSION/src/DQM/CSCMonitorModule
doxygen doc/alldqm.cfg
doxygen doc/cscdqm.cfg
doxygen doc/renderplugin.cfg

