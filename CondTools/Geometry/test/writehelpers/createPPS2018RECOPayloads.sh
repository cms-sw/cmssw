#!/bin/sh

if [ $# -ne 1 ]
then
  echo Error: createPPS2018RECOPayloads.sh requires exactly one argument which is the tag
  exit 1
fi
mytag=$1
echo ${mytag}

# Set the tag in all the scripts and the metadata text files
sed -i {s/TagXX/${mytag}/g} geometryCTPPS2018_writer.py
sed -i {s/TagXX/${mytag}/g} geometryCTPPS2018*.txt
sed -i {s/TagXX/${mytag}/g} splitPPS2018Database.sh

# First read in the little XML files and create the
# Input cff                                             Output file
# geometryRPFromDD_2018_cfi                 ge2018SingleBigFile.xml
cmsRun geometryCTPPS2018_xmlwriter.py

# Now convert the content of the large XML file into
# a "blob" and write it to the database.
# Also reads in the little XML files again and fills
# the DDCompactView. From the DDCompactView the
# reco parts of the database are also filled.
cmsRun geometryCTPPS2018_writer.py

# All the database objects were written into one database
# (myfile.db) in the steps above.  Extract the different
# pieces into separate database files.  These are the payloads
# that get uploaded to the dropbox.  There is one for each tag
./splitPPS2018Database.sh
