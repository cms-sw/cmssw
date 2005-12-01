#!/bin/csh

setenv DB_OWN CMS_VAL_DT_POOL_OWNER@cms_val_lb.cern.ch/val_dt_own_1031

sqlplus ${DB_OWN} << EOF | grep ENTRY | grep -vi COLUMN_NAME | awk -f dtt0mod.awk | sqlplus ${DB_OWN}
set line 20000;
set pagesize 200;
select 'ENTRY',POOL_OR_MAPPING_COLUMNS.SCOPE_NAME||'::',
               POOL_OR_MAPPING_COLUMNS.VARIABLE_NAME,
               POOL_OR_MAPPING_COLUMNS.COLUMN_NAME,
               POOL_OR_MAPPING_ELEMENTS.TABLE_NAME from
               POOL_OR_MAPPING_COLUMNS,
               POOL_OR_MAPPING_ELEMENTS where
               POOL_OR_MAPPING_COLUMNS.SCOPE_NAME=
               POOL_OR_MAPPING_ELEMENTS.SCOPE_NAME and
               POOL_OR_MAPPING_COLUMNS.VARIABLE_NAME=
               POOL_OR_MAPPING_ELEMENTS.VARIABLE_NAME;
EOF
