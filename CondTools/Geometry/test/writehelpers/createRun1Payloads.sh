#!/bin/sh

if [ $# -ne 1 ]
then
  echo Error: createRun1Payloads.sh requires exactly one argument which is the tag
  exit 1
fi
mytag=$1
echo ${mytag}

# Set the tag in all the scripts and the metadata text files
sed -i {s/TagXX/${mytag}/g} *.py
sed -i {s/TagXX/${mytag}/g} *.txt
sed -i {s/TagXX/${mytag}/g} splitRun1Database.sh

# First read in the little XML files and create the
# large XML file for the Extended2015 scenario.
# Input cff                    Output file
# GeometryExtended_cff         geSingleBigFile.xml
cmsRun geometryxmlwriter.py

# Now convert the content of the large XML file into
# a "blob" and write it to the database.
# Also reads in the little XML files again and fills
# the DDCompactView. From the DDCompactView the
# reco parts of the database are also filled.
cmsRun geometrywriter.py

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
sed -i '{s/Extended/ExtendedZeroMaterial/g}' geometryxmlwriter.py
sed -i '{s/\/ge/\/gez/g}' geometryxmlwriter.py
cmsRun geometryxmlwriter.py
sed -i '{s/ExtendedZeroMaterial/Ideal/g}' geometryxmlwriter.py
sed -i '{s/\/gez/\/gi/g}' geometryxmlwriter.py
cmsRun geometryxmlwriter.py

# Read the one big XML file and output a record to the
# database with the an identifying tag
# This is repeated several times below.  The sed commands
# serve to give the following sequence of input file and output
# tag
#
# Input file                Output tag
# gegSingleBigFile.xml      XMLFILE_Geometry_${mytag}_Extended2015GFlash_mc
# giSingleBigFile.xml       XMLFILE_Geometry_${mytag}_Ideal_mc
#
sed -i '{s/Extended/ExtendedZeroMaterial/g}' xmlgeometrywriter.py
sed -i '{s/\/ge/\/gez/g}' xmlgeometrywriter.py
cmsRun xmlgeometrywriter.py
sed -i '{s/ExtendedZeroMaterial/Ideal/g}' xmlgeometrywriter.py
sed -i '{s/\/gez/\/gi/g}' xmlgeometrywriter.py
cmsRun xmlgeometrywriter.py

# All the database objects were written into one database
# (myfile.db) in the steps above.  Extract the different
# pieces into separate database files.  These are the payloads
# that get uploaded to the dropbox.  There is one for each tag
./splitRun1Database.sh
