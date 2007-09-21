#!/bin/sh

# When using compressed LUTs, prepare each XML file using this command 


for file in $*
do
echo $file
gzip -c $file | uuencode -m $file.gz > $file.gz.uue

done
#gzip -c [file.xml] | uuencode -m LUT > [file.xml.gz.uue]

