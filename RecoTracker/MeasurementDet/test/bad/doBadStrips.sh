#!/bin/bash
rm dbfile-badstrip.db
cmscond_bootstrap_detector.pl --offline_connect sqlite_file:dbfile-badstrip.db --auth=/afs/cern.ch/cms/DB/conddb/authentication.xml STRIP
perl mkBadStrips.pl
cmsRun SiStripBadStripBuilder.cfg 2>&1 | tee SiStripBadStripBuilder.log 
