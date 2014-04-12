#!/bin/bash
rm dbfile.db
cmscond_bootstrap_detector.pl --offline_connect sqlite_file:dbfile.db --auth=/afs/cern.ch/cms/DB/conddb/authentication.xml STRIP
cat > BadAPVs.cff <<_EOF_
replace prod.BadComponentList = { 
    { uint32 BadModule = 470178036 vuint32 BadApvList = {0,1} }
}
_EOF_
cat > BadModules.cff  <<_EOF_
replace prod.BadModuleList = {
    470178036
}
_EOF_
cat > BadStrips.cff <<_EOF_
replace prod.BadComponentList = {
   { uint32 BadModule = 470178036 vuint32 BadChannelList = { 0 } }
}
_EOF_

rm dbfile-badapv.db; 
mv dbfile.db dbfile-badapv.db;
cmsRun SiStripBadAPVBuilder.cfg 2>&1 | tee SiStripBadAPVBuilder.log
mv dbfile-badapv.db dbfile.db;

rm dbfile-badstrip.db;
mv dbfile.db dbfile-badstrip.db;
cmsRun SiStripBadStripBuilder.cfg 2>&1 | tee SiStripBadStripBuilder.log
mv dbfile-badstrip.db dbfile.db;

rm dbfile-badmodule.db;
mv dbfile.db dbfile-badmodule.db;
cmsRun SiStripBadModuleBuilder.cfg 2>&1 | tee SiStripBadModuleBuilder.log
mv dbfile-badmodule.db dbfile.db;

echo "Created an 'almost ideal' DB with all ok except module 470178036"
