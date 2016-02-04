#! /bin/bash

##
## $1 = list of ALCARECO files to process
## $2: label to append to files in output (optional)


if [ "$#" == 0 ]; then
   echo "Usage: cntevts_in_file.sh <filelist>"
   exit 1
fi

TAG="nevents${2}"

rm -f $TAG.*

for file in $( cat $1 )
  do
  echo
  echo "Querying DBS for $file"
  echo "---> $(./query.py --input="find file.numevents where file=$file"  --verbose=0 --limit=-1)" >> $TAG".tmp"
####--limit=-1 
  echo "#events: $( tail -n 2 $TAG.tmp )"
done



IND=0
takenext=0

for line in $(cat $TAG".tmp")
do
let IND=IND+1

if [ $IND == 1 ] 
then
echo
fi

if [ $takenext -ge 1 ] 
then
let takenext=takenext+1
fi


#if [ "$line" == "file.numeventss" ] 
if [ "$line" == "Found" ] 
then
#echo "~~~~LINE is $line"
#echo "~~~~IND is $IND"
takenext=1
fi

#the number of events appears as 3rd field after the 
#word 'Found'

if [ $takenext == 4 ] 
then
IND=0
###if [ $line -gt 0 ] #don't include zero events ALCARECO
###    then
    echo $line >> ${TAG}".out"
###fi
takenext=0
fi

done

mv ${TAG}".out" ../data/
#rm -f ${TAG}".tmp"
