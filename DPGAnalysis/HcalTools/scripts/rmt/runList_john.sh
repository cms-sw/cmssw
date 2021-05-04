#!/bin/bash
# John Hakala, 9/14/15
# To use this script, run it like this:
# ./runList.sh <lower limit run number> <upper limit run number>
# Example:
# ./runList.sh 255000 255200
#
# It must be run for the cmsusr network, or a tunnel must be set up to forward the query

# Build an sql script for checking whether the run is local or global
echo -n 'SELECT  "STRING_VALUE" FROM "CMS_RUNINFO"."RUNSESSION_PARAMETER" WHERE "RUNNUMBER"=' > checkGlobal.sql
echo -n "'&1'"  >> checkGlobal.sql
echo -n 'AND "NAME" LIKE'  >> checkGlobal.sql
echo  "'%RUN_TYPE';" >> checkGlobal.sql
echo "exit;" >> checkGlobal.sql

# Start building an sql query for dumping the desired info from local runs
# All these first lines are just to make the output pretty
echo 'set linesize 300;' > tmp.sql
echo 'set wrap off;' >> tmp.sql
echo 'set trimout on;' >> tmp.sql
echo 'set tab off;' >> tmp.sql
echo 'column NAME format a30' >> tmp.sql
echo 'column TIME format a50' >> tmp.sql
echo 'column STRING_VALUE format a80' >> tmp.sql

# Loop over user-specified run-number range
for ((i=$1; i<=$2; i++))
do 
	# Use the script for checking if a run is global or local
	sqlplus cms_hcl_runinfo/run2009info@cms_omds_adg @checkGlobal.sql $i | grep -q GLOBAL
	# If it's global, then do nothing
	if [ "$?" = "0" ]; then
		echo -n
        # Otherwise, add the local run number to the query for looking at local runs.
	else
		echo -n 'SELECT  "RUNNUMBER","TIME", "NAME","STRING_VALUE" FROM "CMS_RUNINFO"."RUNSESSION_PARAMETER" WHERE "RUNNUMBER"=' >> tmp.sql
		echo -n $i >> tmp.sql
		echo -n ' AND ("NAME" LIKE' >> tmp.sql
		echo -n " '%FULLPATH'" >> tmp.sql
		echo -n ' OR "NAME" LIKE ' >> tmp.sql
		echo -n "'%TRIGGERS%'" >> tmp.sql
		echo ');' >> tmp.sql
	fi
done
# Finish building the query for all the local runs
echo 'exit;' >> tmp.sql

# Execute the query to dump the desired info for the local runs
sqlplus cms_hcl_runinfo/run2009info@cms_omds_adg @tmp.sql | grep -v "rows selected" | cat -s

