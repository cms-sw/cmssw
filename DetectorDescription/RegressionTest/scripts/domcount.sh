#! /bin/csh -f
eval `scramv1 runtime -csh`
grep ".xml" $CMSSW_RELEASE_BASE/src/Geometry/CMSCommonData/python/cmsIdealGeometryXML_cfi.py | sed "{s/'//g}" | sed '{s/,//g}' | sed '{s/ //g}' | sed '{s/\t//g}' | sed '{s/geomXMLFiles=cms.vstring(//g}' | sed '{s/)//g}' | grep -v "#" >! /tmp/tmpcmsswdddxmlfileslist
cd $CMSSW_RELEASE_BASE/src
DOMCount -v=always -n -s -l /tmp/tmpcmsswdddxmlfileslist
