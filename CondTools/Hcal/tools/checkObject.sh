#!/bin/ksh
TEMPFILE="cmdchk_TEMP.txt"
JOBFILESTEM="checkObject"

if [ "$1" = "" ] ; then
  echo "Usage: cmd_checkObject.sh <listOfOperations.txt> <mod-file>"
  exit 1
fi

CMDFILE=$1

if [ "$2" = "" ] ; then
  MODFILE=`grep '#MODFILE' $CMDFILE | awk '{print $2}'`
else
  MODFILE=$2
fi

echo ----- check object script -----
echo Input file is $CMDFILE
echo Modfile is $MODFILE

echo $TEMPFILE
grep -v '#' $CMDFILE | grep -v '^$' > $TEMPFILE

echo --------- begin list -----
cat $TEMPFILE
echo --------- end list -----

CHOBJECTLIST=`cat $TEMPFILE | awk '{print $1}'`
CHREFFILELIST=`cat $TEMPFILE | awk '{print $2}'`
CHUPFILELIST=`cat $TEMPFILE | awk '{print $3}'`
CHOUTFILELIST=`cat $TEMPFILE | awk '{print $4}'`

jobnumber=0
for myobject in `echo $CHOBJECTLIST`
do
    CHOBJECT=${myobject}
    JOBFILE=${JOBFILESTEM}_${jobnumber}.cfg
# now make the cfg-file and replace <Object> by myobject:
    sed "s|<Object>|$CHOBJECT|g" \
       $MODFILE > $JOBFILE
    jobnumber=`expr $jobnumber + 1`
done

# now include files:
# reference:
jobnumber=0
for myfile in `echo $CHREFFILELIST`
do
    CHREFFILE=${myfile}
    JOBFILE=${JOBFILESTEM}_${jobnumber}.cfg
    sed -i "s|<RefFilename>|$CHREFFILE|g" \
	$JOBFILE
    jobnumber=`expr $jobnumber + 1`
done

# update:
jobnumber=0
for myfile in `echo $CHUPFILELIST`
do
    CHUPFILE=${myfile}
    JOBFILE=${JOBFILESTEM}_${jobnumber}.cfg
    sed -i "s|<UpFilename>|$CHUPFILE|g" \
	$JOBFILE
    jobnumber=`expr $jobnumber + 1`
done

# output:
jobnumber=0
for myfile in `echo $CHOUTFILELIST`
do
    CHOUTFILE=${myfile}
    JOBFILE=${JOBFILESTEM}_${jobnumber}.cfg
    sed -i "s|<OutFilename>|$CHOUTFILE|g" \
	$JOBFILE
    jobnumber=`expr $jobnumber + 1`
done

echo produced:
ls -l ${JOBFILESTEM}_*.cfg
echo ----------
echo running CMSSW...
echo ----------

#now run it and delete the cfg's
jobnumber=0
for myfile in `echo $CHOBJECTLIST`
do
    JOBFILE=${JOBFILESTEM}_${jobnumber}.cfg
    cmsRun $JOBFILE
    rm -r $JOBFILE
    jobnumber=`expr $jobnumber + 1`
done

 rm -r ${TEMPFILE}
