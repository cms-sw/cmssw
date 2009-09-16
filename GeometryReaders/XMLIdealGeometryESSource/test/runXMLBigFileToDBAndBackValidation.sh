#!/bin/tcsh 
# Michael Case 2009-09-17:  This is meant to validate that the geometry going into the db is
# indeed the same as the geometry that comes from the many xml files which created the db.
# It does NOT check SpecPars (yet).
# It checks the position of all parts in the hierarchy of the graph of parts positioned
# in the detector and is currently (in the file testCompareDumpFiles.py) set to look
# for differences exceeting .0004 mm in x, y and z and .0004 in the elements of the
# rotation matrix.
#
# To run this file, ./runXMLBigFileToDBAndBackValidation.sh in 
# GeometryReaders/XMLIdealGeometryESSource/test
# To RERUN the test, rm -rf workarea.
echo start
cmsenv
mkdir workarea
cd workarea
mkdir db
mkdir xml
cd db
# The rm lines can be removed for debugging to check what is going on.
#rm myfile.db
#rm trXMLFromDB.out
#rm twLoadDBWithXML.out
#rm *.log.xml
#rm *.log
#rm dumpBDB
#rm dumpSpecsdumpBDB
echo start write Ideal
# At this point, I'm preparing the database.
source $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/blob_preparation.txt > twLoadDBWithXML.out
# At this point I'm writing the XML file, 'fred.xml'
cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/geometryxmlwriter.py .
sed -i '{s/GeometryExtended/GeometryIdeal/}' geometryxmlwriter.py >> twLoadDBWithXML.out
cmsRun geometryxmlwriter.py >> twLoadDBWithXML.out
# At this point I'm writing ALL geometries, not only the "big file"
cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/geometrywriter.py .
sed -i '{s/GeometryExtended/GeometryIdeal/}' geometrywriter.py >> twLoadDBWithXML.out
cmsRun geometrywriter.py >> twLoadDBWithXML.out
echo end write Ideal
echo start DB read Ideal
cmsRun ../../testReadXMLFromDB.py > trXMLFromDB.out
echo done with read DB Ideal
cd ../xml
#uncomment for debugging.
#rm trIdeal.out
#rm dumpSTD
#rm dumpSpecdumpSTD
#rm diffgeomIdeal.out
echo start XML read Ideal Geometry
cmsRun ../../readIdealAndDump.py
echo end XML read Ideal Geometry
cd ../..
cmsRun testCompareDumpFiles.py
echo ALL DONE!


