#!/bin/ksh

TEMPFILE="cmdDB_TEMP.txt"
JOBFILESTEM="jobDBwriteObject"

if [ "$1" = "" ] ; then
  echo "Usage: DBwriteObject.sh <listOfOperations.txt> [<mod-file>]"
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

DBWHICHLIST=`cat $TEMPFILE | awk '{print $1}'`
DBOBJECTLIST=`cat $TEMPFILE | awk '{print $2}'`
DBPUTFILELIST=`cat $TEMPFILE | awk '{print $3}'`
DBPUTTAGLIST=`cat $TEMPFILE | awk '{print $4}'`
DBSINCETILLLIST=`cat $TEMPFILE | awk '{print $5}'`
DBTIMELIST=`cat $TEMPFILE | awk '{print $6}'`

jobnumber=0
for item in `echo $DBOBJECTLIST`
do
    JOBFILE=${JOBFILESTEM}_${jobnumber}.cfg
    sed "s|<Object>|${item}|g" $MODFILE > $JOBFILE
    jobnumber=`expr $jobnumber + 1`
done
jobnumber=0
for item in `echo $DBPUTFILELIST`
do
    JOBFILE=${JOBFILESTEM}_${jobnumber}.cfg
    sed -i "s|<PutFilename>|${item}|g" $JOBFILE
    jobnumber=`expr $jobnumber + 1`
done
jobnumber=0
for item in `echo $DBPUTTAGLIST`
do
    JOBFILE=${JOBFILESTEM}_${jobnumber}.cfg
    sed -i "s|<PutTag>|${item}|g" $JOBFILE
    jobnumber=`expr $jobnumber + 1`
done
jobnumber=0
for item in `echo $DBTIMELIST`
do
    JOBFILE=${JOBFILESTEM}_${jobnumber}.cfg
    sed -i "s|<IOVTime>|${item}|g" $JOBFILE
    jobnumber=`expr $jobnumber + 1`
done

# now for the tags:
jobnumber=0
for item in `echo $DBWHICHLIST`
do
    JOBFILE=${JOBFILESTEM}_${jobnumber}.cfg
    if [ "${item}" = "sqlite" ] ; then
	sed -i "s|\#<SQLITE>||" $JOBFILE
    fi
    if [ "${item}" = "orcoff" ] ; then
	sed -i "s|\#<ORCOFF>||" $JOBFILE
    fi
    if [ "${item}" = "orcon" ] ; then
	sed -i "s|\#<ORCON>||" $JOBFILE
    fi
    jobnumber=`expr $jobnumber + 1`
done
jobnumber=0
for item in `echo $DBSINCETILLLIST`
do
    JOBFILE=${JOBFILESTEM}_${jobnumber}.cfg
    if [ "${item}" = "since" ] ; then
	sed -i "s|\#<SINCE>||" $JOBFILE
    else
	sed -i "s|\#<TILL>||" $JOBFILE
    fi
    jobnumber=`expr $jobnumber + 1`
done

echo produced:
ls -l ${JOBFILESTEM}_*.cfg
echo ----------
echo running CMSSW...
echo ----------

#now run it and delete the cfg's
jobnumber=0
for item in `echo $DBOBJECTLIST`
do
    JOBFILE=${JOBFILESTEM}_${jobnumber}.cfg
    cmsRun $JOBFILE
    rm -r $JOBFILE
    jobnumber=`expr $jobnumber + 1`
done

rm -r ${TEMPFILE}
