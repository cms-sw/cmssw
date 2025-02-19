#! /bin/csh -f
#
# This script assumes that the schema file DetectorDescription/Schema/DDLSchema.xsd
# is located as it's specified in an xml file, e.g. in a relative to the xml path.
# This is true for the xml files in a release area.
#
# However, if you check out some xml files locally, please, make sure you check out
# the DetectorDescription/Schema locally as well.
#
# Note, this test will fail to find the schema, if an xml file is placed in
# a subdirectory of a data directory.

eval `scramv1 runtime -csh`
grep ".xml" $CMSSW_RELEASE_BASE/src/Geometry/CMSCommonData/python/cmsExtendedGeometryXML_cfi.py | sed "{s/'//g}" | sed '{s/,//g}' | sed '{s/ //g}' | sed '{s/\t//g}' | sed '{s/geomXMLFiles=cms.vstring(//g}' | sed '{s/)//g}' | grep -v "#" >! /tmp/tmpcmsswdddxmlfileslist
cd $CMSSW_RELEASE_BASE/src

set hms = `echo $CMSSW_SEARCH_PATH | awk 'BEGIN{FS=":"}{for (i=1; i<=NF; i++) print $i}'`

set tmpFile=/tmp/tmpcmsswdddxmlfileslistvalid

foreach line( "`cat /tmp/tmpcmsswdddxmlfileslist`" )
 set fileFound=0
 foreach spath( $hms )
  if( ! $fileFound ) then
   set file=$spath/$line
   if( -f $file ) then
    echo $file >> $tmpFile
    set fileFound=1
   endif
  endif 
 end
end

set script=$CMSSW_BASE/bin/$SCRAM_ARCH/DOMCount
if ( ! -f $script ) then
 set script=$CMSSW_RELEASE_BASE/bin/$SCRAM_ARCH/DOMCount
endif

$script -v=always -n -s -f -l $tmpFile

if( -f  $tmpFile ) then
 rm -f $tmpFile
endif

if( -f  /tmp/tmpcmsswdddxmlfileslist ) then
 rm -f /tmp/tmpcmsswdddxmlfileslist
endif
