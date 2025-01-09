#!/bin/sh
function die { echo $1: status $2 ;  exit $2; }

if [ $# -ne 1 ]
then
  echo Error: createExtended2025Payloads.sh requires exactly one argument which is the tag
  exit 1
fi
mytag=$1
echo ${mytag}

# Set the tag in all the scripts and the metadata text files
#sed -i {s/TagXX/${mytag}/g} *.py
compgen -G "*.txt" > /dev/null && sed -i {s/TagXX/${mytag}/g} *.txt
sed -i {s/TagXX/${mytag}/g} splitExtended2025Database.sh

# First read in the little XML files and create the
# big XML file for the Extended2025DD4hep scenario.
cmsRun geometryExtended2025DD4hep_xmlwriter.py || die 'failed geometryExtended2025DD4hep_xmlwriter.py' $?

# Now convert the content of the large XML file into
# a "blob" and write it to the database.
# Also reads in the little XML files again and fills
# the DDCompactView. From the DDCompactView the
# reco parts of the database are also filled.
cmsRun geometryExtended2025DD4hep_writer.py --tag=${mytag} || die 'failed geometryExtended2025DD4hep_writer.py' $?

# Now put the other scenarios into the database.
# Input the many XML files referenced by the cff file and
# output a single big XML file.
# This is repeated several times below.  The sed commands
# serve to give the correct sequence of input and output
# files

#sed -i '{s/ExtendedGeometry2025/ExtendedGeometry2025ZeroMaterial/g}' geometryExtended2025DD4hep_xmlwriter.py
#sed -i '{s/\/ge/\/gez/g}' geometryExtended2025DD4hep_xmlwriter.py
cmsRun geometryExtended2025DD4hep_xmlwriter.py --geom=ExtendedGeometry2025ZeroMaterial --out=gez || die 'failed geometryExtended2025DD4hep_xmlwriter.py ExtendedGeometry2025ZeroMaterial' $?

#sed -i '{s/ExtendedGeometry2025ZeroMaterial/ExtendedGeometry2025FlatMinus05Percent/g}' geometryExtended2025DD4hep_xmlwriter.py
#sed -i '{s/\/gez/\/geFM05/g}' geometryExtended2025DD4hep_xmlwriter.py
cmsRun geometryExtended2025DD4hep_xmlwriter.py --geom=ExtendedGeometry2025FlatMinus05Percent --out=geFM05 || die 'failed geometryExtended2025DD4hep_xmlwriter.py ExtendedGeometry2025FlatMinus05Percent' $?

#sed -i '{s/ExtendedGeometry2025FlatMinus05Percent/ExtendedGeometry2025FlatMinus10Percent/g}' geometryExtended2025DD4hep_xmlwriter.py
#sed -i '{s/\/geFM05/\/geFM10/g}' geometryExtended2025DD4hep_xmlwriter.py
cmsRun geometryExtended2025DD4hep_xmlwriter.py --geom=ExtendedGeometry2025FlatMinus10Percent --out=geFM10 || die 'failed geometryExtended2025DD4hep_xmlwriter.py' $?

#sed -i '{s/ExtendedGeometry2025FlatMinus10Percent/ExtendedGeometry2025FlatPlus05Percent/g}' geometryExtended2025DD4hep_xmlwriter.py
#sed -i '{s/\/geFM10/\/geFP05/g}' geometryExtended2025DD4hep_xmlwriter.py
cmsRun geometryExtended2025DD4hep_xmlwriter.py --geom=ExtendedGeometry2025FlatPlus05Percent --out=geFP05 || die 'failed geometryExtended2025DD4hep_xmlwriter.py ExtendedGeometry2025FlatPlus05Percent' $?

#sed -i '{s/ExtendedGeometry2025FlatPlus05Percent/ExtendedGeometry2025FlatPlus10Percent/g}' geometryExtended2025DD4hep_xmlwriter.py
#sed -i '{s/\/geFP05/\/geFP10/g}' geometryExtended2025DD4hep_xmlwriter.py
cmsRun geometryExtended2025DD4hep_xmlwriter.py --geom=ExtendedGeometry2025FlatPlus05Percent --out=geFP10 || die 'failed geometryExtended2025DD4hep_xmlwriter.py' $?

# Read the one big XML file and output a record to the
# database with the an identifying tag
# This is repeated several times below.  The sed commands
# serve to give the correct sequence of input file and output
# tag
# To start:
# Input file                Output tag
# gezSingleBigFile.xml      XMLFILE_Geometry_${mytag}_Extended2025ZeroMaterial_mc

#sed -i '{s/Extended/Extended2025ZeroMaterial/g}' xmlgeometrywriter.py
#sed -i '{s/\/ge/\/gez/g}' xmlgeometrywriter.py
cmsRun xmlgeometrywriter.py --tag=${mytag} --out=Extended2025ZeroMaterial --inPre=gez|| die 'failed xmlgeometrywriter.py Extended2025ZeroMaterial' $?

#sed -i '{s/Extended2025ZeroMaterial/Extended2025FlatMinus05Percent/g}' xmlgeometrywriter.py
#sed -i '{s/\/gez/\/geFM05/g}' xmlgeometrywriter.py
cmsRun xmlgeometrywriter.py --tag=${mytag} --out=Extended2025FlatMinus05Percent --inPre=geFM05 || die 'failed xmlgeometrywriter.py Extended2025FlatMinus05Percent' $?

#sed -i '{s/Extended2025FlatMinus05Percent/Extended2025FlatMinus10Percent/g}' xmlgeometrywriter.py
#sed -i '{s/\/geFM05/\/geFM10/g}' xmlgeometrywriter.py
cmsRun xmlgeometrywriter.py --tag=${mytag} --out=Extended2025FlatMinus10Percent --inPre=geFM10 || die 'failed xmlgeometrywriter.py Extended2025FlatMinus10Percent' $?

#sed -i '{s/Extended2025FlatMinus10Percent/Extended2025FlatPlus05Percent/g}' xmlgeometrywriter.py
#sed -i '{s/\/geFM10/\/geFP05/g}' xmlgeometrywriter.py
cmsRun xmlgeometrywriter.py --tag=${mytag} --out=Extended2025FlatPlus05Percent --inPre=geFP05 || die 'failed xmlgeometrywriter.py Extended2025FlatPlus05Percent' $?

#sed -i '{s/Extended2025FlatPlus05Percent/Extended2025FlatPlus10Percent/g}' xmlgeometrywriter.py
#sed -i '{s/\/geFP05/\/geFP10/g}' xmlgeometrywriter.py
cmsRun xmlgeometrywriter.py --tag=${mytag} --out=Extended2025FlatPlus10Percent --inPre=geFP10 || die 'failed xmlgeometrywriter.py Extended2025FlatPlus10Percent' $?

# All the database objects were written into one database
# (myfile.db) in the steps above.  Extract the different
# pieces into separate database files.  These are the payloads
# that get uploaded to the DB.  There is one for each tag
./splitExtended2025Database.sh
