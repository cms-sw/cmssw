#!/bin/sh

workdir=`dirname $0`
cd $workdir

if [ "c${CMSSW_RELEASE_BASE}" == "c" ]; then
    echo -e "\nSetting scramv1 runtime"
    cd /afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/cmssw/CMSSW_1_6_0
    eval `scramv1 runtime -sh`
    cd -
fi
export TNS_ADMIN=/afs/cern.ch/project/oracle/admin
path=/afs/cern.ch/cms/DB/conddb/*.xml
#path=./conddb/*.xml

sqlite3 dbfile.db < CreateSqliteTable.sql

#grep -i "connection name"  $path | grep -i strip | awk  -F"\"" '{print $2}' | grep -v frontier | grep -v AUDIT_TEST | sort | uniq 


#get connection string
for connection in `grep -i "connection name"  $path | grep -i strip | awk  -F"\"" '{print $2}' | grep -v frontier | grep -v AUDIT_TEST | sort | uniq `
  do
  #echo -e "\n$connection"

  user=`grep -A3 $connection $path | head -3 | grep -v "connection name" | awk '$0~/name=\"user\"/{print $0}' | awk -F ' <parameter name="user" value=' '{print $2}' | awk -F'\"' '{print $2}' `

 pass=`grep -A3 $connection $path | head -3 | grep -v "connection name" | awk '$0~/name=\"password\"/{print $0}' | awk -F ' <parameter name="password" value=' '{print $2}' | awk -F'\"' '{print $2}' `

 type=`echo $connection | awk -F":" '{print $1}'`
 db=`echo $connection | awk -F"/" '{print $3}' `
 account=`echo $connection | awk -F"/" '{print $4}' `

 #echo $user  $pass $type $db $account
 #echo "\"select name from $account.metadata;\" |  sqlplus -S $user/$pass@$db | grep -i strip"

 for tag in `echo "select name from $account.metadata;" | sqlplus -S $user/$pass@$db | grep -i strip`
   do
   #echo $tag
   echo "INSERT INTO DBtags values('$db' , '$account' , '$tag');" | sqlite3 dbfile.db
 done
done

./QuerySqlite.sh
