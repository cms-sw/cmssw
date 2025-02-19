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

# What I want to know is 1 where the config file is located (for domcount and dddreport)
# and 2 what the sub-string corresponding to that is in the Configuration/StandardSequences.
cmsenv
if ($#argv == 0) then                                                                                                                                                                                    
   setenv geometry "GeometryIdeal"                                                                                                                                                                      
else                                                                                                                                                                                                     
   setenv geometry `echo ${1}`                                                                                                                                                                          
endif                                                                                                                                                                                                    
                                                                                                                                                                                                        
set geomtemp = `(grep "Geometry.CMSCommonData" ${CMSSW_RELEASE_BASE}/src/Configuration/StandardSequences/python/${geometry}_cff.py | awk 'split($2,a,"."){print a[3]}')` 
#awk -F\. '{print $3}')`
set geomxml = "${CMSSW_RELEASE_BASE}/src/Geometry/CMSCommonData/python/${geomtemp}.py"

echo "START - All messages in this script pertain to geometry data described in Configuration/StandardSequence/python/${geometry}_cff.py"
echo "        and xml files in: ${geomxml}" 

# STEP 1:
# validate current set of xml files in $geomxml is valid
#ASSUMPTIONS:  1.  relative path between documents (xml) and schema (DetectorDescription/Schema/DDLSchema.xsd)
#                  are no more than 4 away, i.e. ../../../../ MAX (for normal/cmsextent.xml files)
#grep ".xml" $geomxml | sed "{s/'//g}" | sed '{s/,//g}' | sed '{s/ //g}' | sed '{s/\t//g}' | sed '{s/geomXMLFiles=cms.vstring(//g}' | sed '{s/)//g}' | grep -v "#" >! /tmp/tmpcmsswdddxmlfileslist
set whst=`(grep ".xml" $geomxml | sed "{s/'//g}" | sed '{s/,//g}' | sed '{s/ //g}' | sed '{s/\t//g}' | sed '{s/geomXMLFiles=cms.vstring(//g}'  | sed '{s/+cms.vstring(//g}' | sed '{s/)//g}' | grep -v "#" )`
#echo $whst
mkdir workarea
#rm -f dcorig.out
touch dcorig.out
#set the schema path
if ( -e "${CMSSW_BASE}/src/DetectorDescription/Schema/DDLSchema.xsd" ) then
    set schpath = `(echo "file://${CMSSW_BASE}/src/DetectorDescription/Schema/DDLSchema.xsd")`
else
    set schpath = `(echo "file://${CMSSW_RELEASE_BASE}/src/DetectorDescription/Schema/DDLSchema.xsd")`
endif
echo "Assuming the schema is here: " $schpath
#prep schpath for feeding into sed.
set schpath = `(echo $schpath | sed '{s/\//\\\//g}')`

    foreach l ( $whst )
	if ( -e $CMSSW_BASE/src/$l ) then
	    set dp = `(echo "${l}" | awk -F\/ '{print NF}')`
	    set fn = `(echo "${l}" | awk -F\/ '{print $NF}')`
	    cp $CMSSW_BASE/src/$l .
	    if ( $dp > 5 ) then
		echo "ERROR: file " $fn " has a relative path too big for this script." 
	    else
		sed -i "{s/..\/..\/..\/..\/DetectorDescription\/Schema\/DDLSchema.xsd/${schpath}/g}" $fn
	    endif
	    sed -i "{s/..\/..\/..\/DetectorDescription\/Schema\/DDLSchema.xsd/${schpath}/}" $fn
	    DOMCount -v=always -n -s -f $fn >>& dcorig.out
	    rm -f $fn
	else
	    if ( -e $CMSSW_RELEASE_BASE/src/$l ) then
		set dp = `(echo "${l}" | awk -F\/ '{print NF}')`
		set fn = `(echo "${l}" | awk -F\/ '{print $NF}')`
		cp $CMSSW_RELEASE_BASE/src/$l .
		if ( $dp > 5 ) then
		    echo "ERROR: file " $fn " has a relative path too big for this script." 
		else
		    sed -i "{s/..\/..\/..\/..\/DetectorDescription\/Schema\/DDLSchema.xsd/${schpath}/g}" $fn
		endif
		sed -i "{s/..\/..\/..\/DetectorDescription\/Schema\/DDLSchema.xsd/${schpath}/}" $fn
		DOMCount -v=always -n -s -f $fn >>& dcorig.out
		rm -f $fn
	    else
		echo "ERROR: file " $l " not found in " $CMSSW_RELEASE_BASE "/src or " $CMSSW_BASE "/src" >>& dcorig.out
	    endif
	endif
    end
    set errcnt = `(grep --count "Error" dcorig.out)`
    set warcnt = `(grep --count "Warning" dcorig.out)`
    if ($errcnt != 0 || $warcnt != 0) then
	echo "WARNING: There ARE XML Schema violations in original XML files and can be seen in dcorig.out."
    else
	echo "There ARE NO XML Schema violations in original XML files."
    endif
#else
#    echo "Missing ../../../DetectorDescription/Schema/DDLSchema.xsd..."
#    echo "If you are running in your own work area, please check out (addpkg) DetectorDescription/Schema in your src directory."
#    echo "ERROR: DOMCount validation not performed... others might still work below."
#endif

cd workarea

# STEP 2:
# validate current ddd model has no missing solids, materials or logical parts
#dddreport.sh >& dddreport.out
cp ${CMSSW_RELEASE_BASE}/test/${SCRAM_ARCH}/DDErrorReport ${CMSSW_BASE}/bin/${SCRAM_ARCH}/.
#ls ${CMSSW_BASE}/bin/${SCRAM_ARCH}/
rehash
../testdddreport.sh ${geomtemp}.py >& dddreport.out
set whst=`(grep -n "Start checking" dddreport.out | awk -F : '{print $1}')`
set totsiz=`(wc -l dddreport.out | awk '{print $1}')`
@ tsdif = $totsiz - $whst
#set tsdif = "${totsiz} - ${whst}"
#echo "GOT HERE " $totsiz " - " $whst 
tail -$tsdif dddreport.out >& dddreptail.out
set diffout = `(diff dddreptail.out ../dddreptail.ref)`
if ( "$diffout" != "") then
    echo "WARNING: There ARE differences in the DD named objects from the standard xml files since the last ddreport.sh was run."
    echo "WARNING: Please check workarea/dddreport.out and workarea/dddreptail.out."
else 
    echo "There ARE NO differences in the DD named objects from the standard xml files since the last ddreport.sh was run."
endif

mkdir db
mkdir xml
cd db
# STEP 3: prepare database, prepare XML file to be loaded in DB.
# The rm lines can be removed for debugging to check what is going on.
#rm -f myfile.db
#rm -f trXMLFromDB.out
#rm -f twLoadDBWithXML.out
#rm -f *.log.xml
#rm -f *.log
#rm -f dumpBDB
#rm -f dumpSpecsdumpBDB
echo "Start to write the single BIG XML file."
# At this point I'm writing the XML file, 'fred.xml'
# ASSUMPTION:  1) In the file CondTools/Geometry/test/geometryxmlwriter.py there will be always GeometryExtended
#                 in the name of the config to be loaded. IF NOT, let me know and I'll adjust this (Mike Case)
touch twLoadDBWithXML.out
cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/geometryxmlwriter.py .
sed -i "{s/GeometryExtended/${geometry}/}" geometryxmlwriter.py >>& twLoadDBWithXML.out
sed -i "{s/geTagXX/fred/g}" geometryxmlwriter.py
#sed -i '{s/GeometryExtended/GeometryIdeal/}' geometryxmlwriter.py >> twLoadDBWithXML.out
cmsRun geometryxmlwriter.py >>& twLoadDBWithXML.out
echo "Finish the write to the single BIG XML file."

# STEP 4:
# make sure fred.xml has appropriate relative path to DDLSchema.xsd

#sed -i '{s/..\/..\/..\/D/..\/..\/..\/..\/..\/D/g}' fred.xml
sed -i "{s/..\/..\/..\/DetectorDescription\/Schema\/DDLSchema.xsd/${schpath}/}" fred.xml
DOMCount -n -s -f -v=always fred.xml >& dcBig.out
#set diffout = `(diff diffdom.out ../../domcountBIG.ref)`
set errcnt = `(grep --count "Error" dcBig.out)`
set warcnt = `(grep --count "Warning" dcBig.out)`
#if ( "$diffout" != "" ) then
if ($errcnt != 0 || $warcnt != 0) then
    echo "WARNING: There ARE Schema violations in the single BIG XML file."
else 
    echo "There ARE NO Schema violations in the single BIG XML file."
endif
# validate current ddd model AS TRANSFERRED TO THE BIG XML FILE has no missing solids, materials or logical parts
grep -v "File" $CMSSW_BASE/src/dddreportconfig.xml >& dddbigfilereport.xml
sed -i '{s/<Root/<File name="GeometryReaders\/XMLIdealGeometryESSource\/test\/workarea\/db\/fred\.xml" url="\."\/><Root/}' dddbigfilereport.xml
DDErrorReport GeometryReaders/XMLIdealGeometryESSource/test/workarea/db/dddbigfilereport.xml >& dddreport.out
set whst=`(grep -n "Start checking" dddreport.out | awk -F : '{print $1}')`
set totsiz=`(wc -l dddreport.out | awk '{print $1}')`
#echo "GOT HERE " $totsiz " - " $whst 
@ tsdif2 = $totsiz - $whst
tail -$tsdif2 dddreport.out >& dddreptail.out
set diffout = `(diff dddreptail.out ../../dddreptail.ref)`
if ( "$diffout" != "" ) then
    echo "WARNING: There ARE differences in the DD named objects from the single BIG xml file since the last ddreport.sh was run.  Please check dddreptail.out."
else 
    echo "There ARE NO differences in the DD named objects from the single BIG xml file since the last ddreport.sh was run."
endif

# STEP 5
echo "Start to write all geometry objects to the local DB including BIG XML file."

# At this point I'm writing ALL geometries, not only the "big file" into the database.
cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/geometrywriter.py .
sed -i "{s/GeometryExtended/${geometry}/}" geometrywriter.py >>& twLoadDBWithXML.out
sed -i "{s/geTagXX/fred/g}" geometrywriter.py >>& twLoadDBWithXML.out
#sed -i '{s/GeometryExtended/GeometryIdeal/}' geometrywriter.py >> twLoadDBWithXML.out
cmsRun geometrywriter.py >>& twLoadDBWithXML.out
echo "Finish writing all geometry objects to the local DB including BIG XML file."
echo "Start to read the big XML file FROM the DB object"
cmsRun ../../testReadXMLFromDB.py >& trXMLFromDB.out
echo "Done with reading the big XML file FROM the DB object"
cd ../xml
#uncomment for debugging.
#rm -f trIdeal.out
#rm -f dumpSTD
#rm -f dumpSpecdumpSTD
#rm -f diffgeomIdeal.out
echo "Start reading the XML from the original config file."
cp ../../readIdealAndDump.py .
sed -i "{s/GeometryExtended/${geometry}/}" readIdealAndDump.py >& trIdeal.out
cmsRun readIdealAndDump.py >>& trIdeal.out
echo "End reading the XML from the original config file."
cd ../..
cmsRun testCompareDumpFiles.py >& tcdf.out
# validate geometry from xml files to db.
set wccnt = `(wc -l tcdf.out | awk '{print $1}')`
if ( $wccnt == 0 ) then
    echo "All differences in position are less than tolerance."
else
    echo "WARNING: There are $wccnt lines with differences greater than tolerance.  Please check tcdf.out for differences."
    echo "WARNING: Tolerance can be changed in the file testCompareDumpFiles.py."
endif

echo "ALL DONE!"


