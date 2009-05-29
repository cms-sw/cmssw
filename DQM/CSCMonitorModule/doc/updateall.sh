#!/bin/bash

cd ~/$CMSSW_VERSION/src/DQM/CSCMonitorModule
doxygen doc/alldqm.doxy
doxygen doc/cscdqm.doxy
doxygen doc/renderplugin.doxy

