#! /bin/tcsh -f
#cmsenv
if ($#argv == 0) then
    setenv geomxml "${LOCAL_TOP_DIR}/src/Geometry/CMSCommonData/python/cmsIdealGeometryXML_cfi.py"
else
    if ($#argv == 1) then
	setenv geomxml `echo ${LOCAL_TOP_DIR}/src/Geometry/CMSCommonData/python/${1}`
    endif
endif
if ( ! -e ${geomxml} ) then 
   echo "ERROR- ${geomxml} file not found"
endif
mkdir -p ./src
echo "START - All messages in this script pertain to geometry data described in xml files in: ${geomxml}" 
echo '<?xml version="1.0"?>' > ./src/dddreportconfig.xml
echo '<Configuration xmlns="http://www.cern.ch/cms/CDL"' >> ./src/dddreportconfig.xml
echo ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"' >> ./src/dddreportconfig.xml
echo ' xsi:schemaLocation= "http://www.cern.ch/cms/CDL ../../../Schema/CDLSchema.xsd"' >> ./src/dddreportconfig.xml
echo ' name="CMSConfiguration" version="0">' >> ./src/dddreportconfig.xml
grep ".xml" $geomxml |  sed "{s/'//g}" | sed '{s/,//g}' | sed '{s/ //g}' | sed '{s/\t//g}' | sed '{s/geomXMLFiles=cms.vstring(//g}' | sed '{s/+cms.vstring(//g}' | sed '{s/)//g}' | grep -v "#" | awk '{print "   <File name=\"" $1 "\" url=\".\"/>"}' >> ./src/dddreportconfig.xml
echo '<Root fileName="cms.xml" logicalPartName="OCMS"/>'  >> ./src/dddreportconfig.xml
echo '<Schema schemaLocation="http://www.cern.ch/cms/DDL  ../../Schema/DDLSchema.xsd" validation="false"/>'  >> ./src/dddreportconfig.xml
echo '</Configuration>'  >> ./src/dddreportconfig.xml
DDErrorReport dddreportconfig.xml
