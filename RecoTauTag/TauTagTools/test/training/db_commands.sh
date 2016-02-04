#!/bin/bash
alias list_prepdb="cmscond_list_iov -c oracle://cms_orcoff_prep/CMS_COND_31X_BTAU -P /afs/cern.ch/cms/DB/conddb -a"
alias list_proddb="cmscond_list_iov -c frontier://FrontierProd/CMS_COND_31X_BTAU -P /afs/cern.ch/cms/DB/conddb -a"
wget http://condb.web.cern.ch/condb/DropBoxOffline/dropBoxOffline_test.sh 
wget http://condb.web.cern.ch/condb/DropBoxOffline/templateForDropbox.txt 


#cmscond_export_iov -s sqlite_file:hpstanc/db/computers.db -d oracle://cms_orcoff_prep/CMS_COND_31X_BTAU -D CondFormatsPhysicsToolsObjects -i Tanc -t hpstanc_v1  -l sqlite:log.db
