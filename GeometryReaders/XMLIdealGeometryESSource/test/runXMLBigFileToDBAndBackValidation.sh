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
echo "START - All messages in this script pertain to geometry data described in cmsIdealGeometryXML_cfi.py"
cmsenv
if ($#argv == 0) then
setenv geometry "GeometryIdeal"
else
setenv geometry `echo ${1}`
endif
echo $geometry
mkdir workarea
cd workarea
# validate current set of xmlfiles in IdealGeometry is correct.
domcount.sh >& dcorig.out
set errcnt = `(grep --count "Error" dcorig.out)`
set warcnt = `(grep --count "Error" dcorig.out)`
if ($errcnt == 0 && $warcnt == 0) then
    echo "No XML Schema violations in original xml files."
else
    echo "XML Schema violations can be seen in dcorig.out."
endif

# validate current ddd model has no missing solids, materials or logical parts
dddreport.sh >& dddreport.out
set whst=`(grep -n "Start checking" dddreport.out | awk -F : '{print $1}')`
set totsiz=`(wc -l dddreport.out | awk '{print $1}')`
@ tsdif = $totsiz - $whst
#echo "GOT HERE " $totsiz " - " $whst 
tail -$tsdif dddreport.out >& dddreptail.out
set diffout = `(diff dddreptail.out ../dddreptail.ref)`
if ( "$diffout" != "") then
    echo "There ARE differences in the DD named objects from the standard xml files since the last ddreport.sh was run.  Please check ddreptail.out."
else 
    echo "There ARE NO differences in the DD named objects from the standard xml files since the last ddreport.sh was run."
endif

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
echo "start write Ideal"
# At this point, I'm preparing the database.
source $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/blob_preparation.txt > twLoadDBWithXML.out
# At this point I'm writing the XML file, 'fred.xml'
cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/geometryxmlwriter.py .
sed -i '{s/GeometryExtended/GeometryIdeal/}' geometryxmlwriter.py >> twLoadDBWithXML.out
cmsRun geometryxmlwriter.py >> twLoadDBWithXML.out

# validate current ddd model AS TRANSFERED TO THE BIG XML FILE has no missing solids, materials or logical parts
grep -v "File" $CMSSW_BASE/src/dddreportconfig.xml > dddbigfilereport.xml
sed -i '{s/<Root/<File name="GeometryReaders\/XMLIdealGeometryESSource\/test\/workarea\/db\/fred\.xml" url="\."\/><Root/}' dddbigfilereport.xml
DDErrorReport GeometryReaders/XMLIdealGeometryESSource/test/workarea/db/dddbigfilereport.xml >& dddreport.out
set whst=`(grep -n "Start checking" dddreport.out | awk -F : '{print $1}')`
set totsiz=`(wc -l dddreport.out | awk '{print $1}')`
#echo "GOT HERE " $totsiz " - " $whst 
@ tsdif2 = $totsiz - $whst
tail -$tsdif2 dddreport.out >& dddreptail.out
set diffout = `(diff dddreptail.out ../dddreptail.ref)`
if ( "$diffout" != "" ) then
    echo "There ARE differences in the DD named objects from the single BIG xml file since the last ddreport.sh was run.  Please check ddreptail.out."
else 
    echo "There ARE NO differences in the DD named objects from the single BIG xml file since the last ddreport.sh was run."
endif

# At this point I'm writing ALL geometries, not only the "big file"
cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/geometrywriter.py .
sed -i '{s/GeometryExtended/GeometryIdeal/}' geometrywriter.py >> twLoadDBWithXML.out
cmsRun geometrywriter.py >> twLoadDBWithXML.out
echo "end write Ideal"
echo "start DB read Ideal"
cmsRun ../../testReadXMLFromDB.py > trXMLFromDB.out
echo "done with read DB Ideal"
cd ../xml
#uncomment for debugging.
#rm trIdeal.out
#rm dumpSTD
#rm dumpSpecdumpSTD
#rm diffgeomIdeal.out
echo "start XML read Ideal Geometry"
cmsRun ../../readIdealAndDump.py
echo "end XML read Ideal Geometry"
cd ../..
cmsRun testCompareDumpFiles.py >& tcdf.out
# validate geometry from xml files to db.
set wccnt = `(wc -l tcdf.out | awk '{print $1}')`
if ( $wccnt == 0 ) then
    echo "All differences in position are less than tolerance."
else
    echo "There are $wccnt lines with differences greater than tolerance.  Please check/verify."
    echo "Tolerance can be changed in the file testCompareDumpFiles.py."
    echo "tcdf.out contains detailed descriptions of differences."
endif

echo "ALL DONE!"


