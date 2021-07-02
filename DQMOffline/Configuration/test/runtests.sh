#!/bin/bash
set -e
set -x

if [[ -z ${LOCAL_TEST_DIR} ]]; then
    LOCAL_TEST_DIR=.
fi

cd $LOCAL_TEST_DIR

DBFILE="sequences.db"
if [[ -n $1 && -n $2 ]]; then
  SECTION="--limit $1 --offset $2"
  DBFILE="sequences$2.db"
  THREADS="--threads 1"
fi

INFILE=""
if [[ -n $3 ]] ; then
  INFILE="--infile $3"
fi

cmsswSequenceInfo.py --runTheMatrix --steps DQM,VALIDATION $INFILE $SECTION --dbfile "$DBFILE" $THREADS
sqlite3 "$DBFILE"  > "legacymodules-${DBFILE}.txt" <<SQL
SELECT edmfamily, edmbase, classname, instancename, step, seqname, wfid 
FROM plugin 
NATURAL JOIN module 
INNER JOIN sequencemodule ON module.id == moduleid 
INNER JOIN sequence ON sequence.id = sequenceid 
NATURAL JOIN workflow 
WHERE edmfamily is null;
SQL

# There are lots of legacy producers and filters, so we check only for analyzers for now.
if grep EDAnalyzer "legacymodules-${DBFILE}.txt" ; then
  echo "There are legacy modules! See list above."
  exit 1
else
  exit 0
fi
