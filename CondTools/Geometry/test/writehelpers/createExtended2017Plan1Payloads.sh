#!/bin/sh


if [ $# -ne 1 ]
then
  echo Error: createExtended2017Plan1Payloads.sh requires exactly one argument which is the tag
  exit 1
fi
mytag=$1
echo ${mytag}

# Set the tag in all the scripts and the metadata text files
sed -i {s/TagXX/${mytag}/g} *.py
compgen -G "*.txt" > /dev/null && sed -i {s/TagXX/${mytag}/g} *.txt
sed -i {s/TagXX/${mytag}/g} splitExtended2017Plan1Database.sh

# First read in the little XML files and create the
# large XML file for the Phase1_R30F12_HCal Ideal scenario.
# Input cff                                             Output file
# GeometryExtended2017Plan1_cff                       geSingleBigFile.xml
cmsRun geometryExtended2017Plan1_xmlwriter.py

# Now convert the content of the large XML file into
# a "blob" and write it to the database.
# Also reads in the little XML files again and fills
# the DDCompactView. From the DDCompactView the
# reco parts of the database are also filled.
cmsRun geometryExtended2017Plan1_writer.py

# Now put the other scenarios into the database.
# Input the many XML files referenced by the cff file and
# output a single big XML file.
# This is repeated several times below.  The sed commands
# serve to give the following sequence of input and output
# files
#
# Input cff                    Output file
# GeometryIdeal_cff            giSingleBigFile.xml
#
sed -i '{s/Extended2017Plan1/Extended2017Plan1ZeroMaterial/g}' geometryExtended2017Plan1_xmlwriter.py
sed -i '{s/\/ge/\/gez/g}' geometryExtended2017Plan1_xmlwriter.py
cmsRun geometryExtended2017Plan1_xmlwriter.py

sed -i '{s/Extended2017Plan1ZeroMaterial/Extended2017Plan1FlatMinus05Perecent/g}' geometryExtended2017Plan1_xmlwriter.py
sed -i '{s/\/gez/\/geFM05/g}' geometryExtended2017Plan1_xmlwriter.py
cmsRun geometryExtended2017Plan1_xmlwriter.py

sed -i '{s/Extended2017Plan1FlatMinus10Percent/Extended2017Plan1FlatPlus05Perecent/g}' geometryExtended2017Plan1_xmlwriter.py
sed -i '{s/\/geFM10/\/geFP05/g}' geometryExtended2017Plan1_xmlwriter.py
cmsRun geometryExtended2017Plan1_xmlwriter.py

sed -i '{s/Extended2017Plan1FlatPlus05Percent/Extended2017Plan1FlatPlus10Perecent/g}' geometryExtended2017Plan1_xmlwriter.py
sed -i '{s/\/geFP05/\/geFP10/g}' geometryExtended2017Plan1_xmlwriter.py
cmsRun geometryExtended2017Plan1_xmlwriter.py

# Read the one big XML file and output a record to the
# database with the an identifying tag
# This is repeated several times below.  The sed commands
# serve to give the following sequence of input file and output
# tag
#
# Input file                Output tag
# gezSingleBigFile.xml      XMLFILE_Geometry_${mytag}_Extended2017Plan1ZeroMaterial_mc
#
sed -i '{s/Extended/Extended2017Plan1ZeroMaterial/g}' xmlgeometrywriter.py
sed -i '{s/\/ge/\/gez/g}' xmlgeometrywriter.py
cmsRun xmlgeometrywriter.py

sed -i '{s/Extended2017ZeroMaterial/Extended2017Plan1FlatMinus05Percent/g}' xmlgeometrywriter.py
sed -i '{s/\/gez/\/geFM05/g}' xmlgeometrywriter.py
cmsRun xmlgeometrywriter.py

sed -i '{s/Extended2017Plan1FlatMinus05Percent/Extended2017Plan1FlatMinus10Percent/g}' xmlgeometrywriter.py
sed -i '{s/\/geFM05/\/geFM10/g}' xmlgeometrywriter.py
cmsRun xmlgeometrywriter.py

sed -i '{s/Extended2017Plan1FlatMinus10Percent/Extended2017Plan1FlatPlus05Percent/g}' xmlgeometrywriter.py
sed -i '{s/\/geFM10/\/geFP05/g}' xmlgeometrywriter.py
cmsRun xmlgeometrywriter.py

sed -i '{s/Extended2017Plan1FlatPlus05Percent/Extended2017Plan1FlatPlus10Percent/g}' xmlgeometrywriter.py
sed -i '{s/\/geFP05/\/geFP10/g}' xmlgeometrywriter.py
cmsRun xmlgeometrywriter.py

# All the database objects were written into one database
# (myfile.db) in the steps above.  Extract the different
# pieces into separate database files.  These are the payloads
# that get uploaded to the dropbox.  There is one for each tag
./splitExtended2017Plan1Database.sh
