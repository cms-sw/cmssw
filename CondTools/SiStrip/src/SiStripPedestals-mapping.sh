#!/bin/tcsh

setenv CORAL_AUTH_USER whoever
setenv CORAL_AUTH_PASSWORD whatever

pool_build_object_relational_mapping -f mapping-template-SiStripPedestals.xml -o SiStripPedestals-mapping-default.xml -b -d CondFormatsSiStripObjects -c sqlite_file:dummy.db -b

rm -f dummy.db
