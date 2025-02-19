#!/bin/bash
rm dbfile-badmodule.db
cmscond_bootstrap_detector.pl --offline_connect sqlite_file:dbfile-badmodule.db --auth=/afs/cern.ch/cms/DB/conddb/authentication.xml STRIP
perl mkBadModules.pl
cmsRun SiStripBadModuleBuilder.cfg 2>&1 | tee SiStripBadModuleBuilder.log 
