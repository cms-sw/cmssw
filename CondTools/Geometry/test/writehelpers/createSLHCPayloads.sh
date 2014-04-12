#!/bin/sh

if [ $# -ne 1 ]
then
  echo Error: createSLHCPayloads.sh requires exactly one argument which is the tag
  exit 1
fi
mytag=$1
echo ${mytag}

# Set the tag in all the scripts and the metadata text files
sed -i {s/TagXX/${mytag}/g} *.py
sed -i {s/TagXX/${mytag}/g} *.txt
sed -i {s/TagXX/${mytag}/g} splitSLHCDatabase.sh

# First read in the little XML files and create the
# large XML file for the Phase1_R30F12_HCal Ideal scenario.
# Input cff                                             Output file
# Phase1_R30F12_HCal_cmsSimIdealGeometryXML_cff         geSingleBigFile.xml
cmsRun geometrySLHC_Phase1_R30F12_HCal_xmlwriter.py

# Now convert the content of the large XML file into
# a "blob" and write it to the database.
# Also reads in the little XML files again and fills
# the DDCompactView. From the DDCompactView the
# reco parts of the database are also filled.
cmsRun geometrySLHC_Phase1_R30F12_HCal_writer.py

# All the database objects were written into one database
# (myfile.db) in the steps above.  Extract the different
# pieces into separate database files.  These are the payloads
# that get uploaded to the dropbox.  There is one for each tag
./splitSLHCDatabase.sh
