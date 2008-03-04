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
#path=/afs/cern.ch/cms/DB/conddb/*.xml
path=./conddb/*.xml

sqlite3 dbfile.db < CreateSqliteTable.sql

#grep -i "connection name"  $path | grep -i '\(strip\)\|\(pixel\)' | awk  -F"\"" '{print $2}' | grep -v frontier | grep -v AUDIT_TEST | sort | uniq 


#get connection string
for connection in `grep -i "connection name"  $path | grep -i '\(strip\)\|\(pixel\)' | awk  -F"\"" '{print $2}' | grep -vi frontier | grep -vi AUDIT_TEST | sort | uniq `
  do
  #echo -e "\n$connection"

  user=(`grep -A3 "$connection\"" $path | grep -v "connection name" | awk '$0~/name=\"user\"/{print $0}' | awk -F ' <parameter name="user" value=' '{print $2}' | awk -F'\"' '{print $2}' `)

 pass=(`grep -A3 "$connection\"" $path  | grep -v "connection name" | awk '$0~/name=\"password\"/{print $0}' | awk -F ' <parameter name="password" value=' '{print $2}' | awk -F'\"' '{print $2}' `)

 type=`echo $connection | awk -F":" '{print $1}'`
 db=`echo $connection | awk -F"/" '{print $3}' `
 account=`echo $connection | awk -F"/" '{print $4}' `

 i=0
 while [ $i -lt ${#user[@]} ];
   do
   auser=${user[$i]}
   apass=${pass[$i]}
 #echo $auser  $apass $type $db $account
   echo -e "\n $auser @ $db"
   #echo "echo \"select name from $account.metadata;\" |  sqlplus -S $auser/$apass@$db "
 
   result=`echo "select name from $account.metadata;" | sqlplus -S $auser/$apass@$db | grep "\([a-Z]\)\|\([0-9]\)" | grep -v "rows selected" | grep -v '^NAME$'`
   #echo $result
   if [ `echo $result | grep -c "ERROR.*ORA-"` -eq 0 ]; then
       for tag in `echo $result | tr " " "\n"`
	 do
       #echo $tag
	 echo "INSERT INTO DBtags values('$db' , '$account' , '$tag');" | sqlite3 dbfile.db
       done
       i=10000000
   else
       echo $result | tr "\t" "\n" 
       let i=$i+1
   fi
 done

done

./QuerySqlite.sh strip
./QuerySqlite.sh pixel
