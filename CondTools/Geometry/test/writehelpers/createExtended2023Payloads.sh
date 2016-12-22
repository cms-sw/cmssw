#!/bin/sh


if [ $# -ne 1 ]
then
  echo Error: createExtended2023Payloads.sh requires exactly one argument which is the tag
  exit 1
fi
mytag=$1
echo ${mytag}

# Set the tag in all the scripts and the metadata text files
sed -i {s/TagXX/${mytag}/g} *.py
sed -i {s/TagXX/${mytag}/g} *.txt
sed -i {s/TagXX/${mytag}/g} splitExtended2023Database.sh

# First read in the little XML files and create the
# large XML file for the Extended2023D1 scenario.
# Input cff                                        Output file
# GeometryExtended2023D1_cff                       geD1SingleBigFile.xml
cmsRun geometryExtended2023_xmlwriter.py

# Now convert the content of the large XML file into
# a "blob" and write it to the database.
# Also reads in the little XML files again and fills
# the DDCompactView. From the DDCompactView the
# reco parts of the database are also filled.
cmsRun geometryExtended2023_writer.py

# Now put the other scenarios into the database.
# Input the many XML files referenced by the cff file and
# output a single big XML file.
# This is repeated several times below.  The sed commands
# serve to give the following sequence of input and output
# files
#
# Input cff                             Output file
# GeometryExtended2023D2_cff            geD2SingleBigFile.xml
# GeometryExtended2023D3_cff            geD3SingleBigFile.xml
# GeometryExtended2023D4_cff            geD4SingleBigFile.xml
# GeometryExtended2023D5_cff            geD5SingleBigFile.xml
# GeometryExtended2023D6_cff            geD6SingleBigFile.xml
#
sed -i '{s/Extended2023D1/Extended2023D2/g}' geometryExtended2023_xmlwriter.py
sed -i '{s/\/geD1/\/geD2/g}' geometryExtended2023_xmlwriter.py
cmsRun geometryExtended2023_xmlwriter.py
sed -i '{s/Extended2023D2/Extended2023D3/g}' geometryExtended2023_xmlwriter.py
sed -i '{s/\/geD2/\/geD3/g}' geometryExtended2023_xmlwriter.py
cmsRun geometryExtended2023_xmlwriter.py
sed -i '{s/Extended2023D3/Extended2023D4/g}' geometryExtended2023_xmlwriter.py
sed -i '{s/\/geD3/\/geD4/g}' geometryExtended2023_xmlwriter.py
cmsRun geometryExtended2023_xmlwriter.py
sed -i '{s/Extended2023D4/Extended2023D5/g}' geometryExtended2023_xmlwriter.py
sed -i '{s/\/geD4/\/geD5/g}' geometryExtended2023_xmlwriter.py
cmsRun geometryExtended2023_xmlwriter.py
sed -i '{s/Extended2023D5/Extended2023D6/g}' geometryExtended2023_xmlwriter.py
sed -i '{s/\/geD5/\/geD6/g}' geometryExtended2023_xmlwriter.py
cmsRun geometryExtended2023_xmlwriter.py

# Read the one big XML file and output a record to the
# database with the an identifying tag
# This is repeated several times below.  The sed commands
# serve to give the following sequence of input file and output
# tag
#
# Input file                Output tag
# geD2SingleBigFile.xml     XMLFILE_Geometry_${mytag}_Extended2023D2_mc
#
sed -i '{s/Extended2023D1/Extended2023D2/g}' geometryExtended2023_xmlgeometrywriter.py
sed -i '{s/\/geD1/\/geD2/g}' geometryExtended2023_xmlgeometrywriter.py
cmsRun geometryExtended2023_xmlgeometrywriter.py
sed -i '{s/Extended2023D2/Extended2023D3/g}' geometryExtended2023_xmlgeometrywriter.py
sed -i '{s/\/geD2/\/geD3/g}' geometryExtended2023_xmlgeometrywriter.py
cmsRun geometryExtended2023_xmlgeometrywriter.py
sed -i '{s/Extended2023D3/Extended2023D4/g}' geometryExtended2023_xmlgeometrywriter.py
sed -i '{s/\/geD3/\/geD4/g}' geometryExtended2023_xmlgeometrywriter.py
cmsRun geometryExtended2023_xmlgeometrywriter.py
sed -i '{s/Extended2023D4/Extended2023D5/g}' geometryExtended2023_xmlgeometrywriter.py
sed -i '{s/\/geD4/\/geD5/g}' geometryExtended2023_xmlgeometrywriter.py
cmsRun geometryExtended2023_xmlgeometrywriter.py
sed -i '{s/Extended2023D5/Extended2023D6/g}' geometryExtended2023_xmlgeometrywriter.py
sed -i '{s/\/geD5/\/geD6/g}' geometryExtended2023_xmlgeometrywriter.py
cmsRun geometryExtended2023_xmlgeometrywriter.py

# All the database objects were written into one database
# (myfile.db) in the steps above.  Extract the different
# pieces into separate database files.  These are the payloads
# that get uploaded to the dropbox.  There is one for each tag
./splitExtended2023Database.sh
