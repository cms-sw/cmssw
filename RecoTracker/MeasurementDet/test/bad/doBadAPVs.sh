#!/bin/bash
rm dbfile-badapv.db
cmscond_bootstrap_detector.pl --offline_connect sqlite_file:dbfile-badapv.db --auth=/afs/cern.ch/cms/DB/conddb/authentication.xml STRIP
perl mkBadAPVs.pl
cmsRun SiStripBadAPVBuilder.cfg 2>&1 | tee SiStripBadAPVBuilder.log 
