#!/bin/bash
set -e
set -x

if [[ -z ${LOCAL_TEST_DIR} ]]; then
    LOCAL_TEST_DIR=.
fi

cmsswSequenceInfo.py --runTheMatrix --steps DQM,VALIDATION
sqlite3 sequences.db  > legacymodules.txt <<SQL
SELECT edmfamily, edmbase, classname, instancename, step, seqname, wfid 
FROM plugin 
NATURAL JOIN module 
INNER JOIN sequencemodule ON module.id == moduleid 
INNER JOIN sequence ON sequence.id = sequenceid 
NATURAL JOIN workflow 
WHERE edmfamily is null;
SQL

# There are lots of legacy producers and filters, so we check only for analyzers for now.
if grep EDAnalyzer legacymodules.txt ; then
  echo "There are legacy modules! See list above."
  exit 1
else
  exit 0
fi
