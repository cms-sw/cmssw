#! /bin/csh -f
eval `scramv1 runtime -csh`
grep ".xml" $CMSSW_RELEASE_BASE/src/Geometry/CMSCommonData/data/cmsIdealGeometryXML.cfi | replace '"' ' ' | replace "," " " | sed '{s/ //g}' | sed '{s/\t//g\
}' | grep -v "#" >! /tmp/tmpcmsswdddxmlfileslist
cd $CMSSW_RELEASE_BASE/src
DOMCount -v=always -n -s -l /tmp/tmpcmsswdddxmlfileslist
