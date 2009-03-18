#!/bin/tcsh

cmscond_bootstrap_detector -D L1T -f $CMSSW_BASE/src/CondTools/L1Trigger/test/dbconfiguration.xml -b $CMSSW_BASE
