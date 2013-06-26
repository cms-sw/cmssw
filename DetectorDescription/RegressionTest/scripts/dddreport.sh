#! /bin/csh -f
eval `scramv1 runtime -csh`
echo '<?xml version="1.0"?>' > $CMSSW_BASE/src/dddreportconfig.xml
echo '<Configuration xmlns="http://www.cern.ch/cms/CDL"' >> $CMSSW_BASE/src/dddreportconfig.xml
echo ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"' >> $CMSSW_BASE/src/dddreportconfig.xml
echo ' xsi:schemaLocation= "http://www.cern.ch/cms/CDL ../../../Schema/CDLSchema.xsd"' >> $CMSSW_BASE/src/dddreportconfig.xml
echo ' name="CMSConfiguration" version="0">' >> $CMSSW_BASE/src/dddreportconfig.xml
grep ".xml" $CMSSW_RELEASE_BASE/src/Geometry/CMSCommonData/python/cmsIdealGeometryXML_cfi.py |  sed "{s/'//g}" | sed '{s/,//g}' | sed '{s/ //g}' | sed '{s/\t//g}' | sed '{s/geomXMLFiles=cms.vstring(//g}' | sed '{s/)//g}' | grep -v "#" | awk '{print "   <File name=\"" $1 "\" url=\".\"/>"}' >> $CMSSW_BASE/src/dddreportconfig.xml
echo '<Root fileName="cms.xml" logicalPartName="OCMS"/>'  >> $CMSSW_BASE/src/dddreportconfig.xml
echo '<Schema schemaLocation="http://www.cern.ch/cms/DDL  ../../Schema/DDLSchema.xsd" validation="false"/>'  >> $CMSSW_BASE/src/dddreportconfig.xml
echo '</Configuration>'  >> $CMSSW_BASE/src/dddreportconfig.xml
cd $CMSSW_RELEASE_BASE/src

set script=$CMSSW_BASE/test/$SCRAM_ARCH/DDErrorReport 
if ( ! -f $script ) then
 set script=$CMSSW_RELEASE_BASE/test/$SCRAM_ARCH/DDErrorReport
endif

echo $script;

$script dddreportconfig.xml


