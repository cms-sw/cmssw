BEGIN {CLASS="DTT0"; \
       CDATA="DTT0::cellData"; \
       SCOPE="DTT0::cellData::DTCellT0Data"}
($2==     "::" && $3==     "DTT0") {OBJECT_TABLE_NAME=$5}
($2==SCOPE"::" && $3==   "cellId") {COLUMN_NAME="CELL"; VECTOR_TABLE_NAME=$5};
($2==SCOPE"::" && $3==  "layerId") {COLUMN_NAME="LAYER"};
($2==SCOPE"::" && $3==     "slId") {COLUMN_NAME="SUPERLAYER"};
($2==SCOPE"::" && $3=="stationId") {COLUMN_NAME="STATION"};
($2==SCOPE"::" && $3== "sectorId") {COLUMN_NAME="SECTOR"};
($2==SCOPE"::" && $3==  "wheelId") {COLUMN_NAME="WHEEL"};
($2==SCOPE"::" && $3==   "t0mean") {COLUMN_NAME="T0_MEAN"};
($2==SCOPE"::" && $3==   "t0rms" ) {COLUMN_NAME="T0_RMS"};
($2==SCOPE"::") {
     print "update POOL_OR_MAPPING_COLUMNS"                    \
           " set COLUMN_NAME=\047"                             \
                 COLUMN_NAME"\047"                             \
           " where VARIABLE_NAME=\047"$3"\047 and"             \
           " SCOPE_NAME=\047"SCOPE"\047;";
     print "alter table "$5                                    \
           " rename column "$4" to "COLUMN_NAME";"}
END {print "update POOL_OR_MAPPING_COLUMNS"                    \
           " set   COLUMN_NAME=\047DATA_ID\047"                \
           " where COLUMN_NAME=\047POS\047 and"                \
           " SCOPE_NAME=\047"CLASS"\047;";
     print "update POOL_OR_MAPPING_COLUMNS"                    \
           " set   COLUMN_NAME=\047DATA_ID\047"                \
           " where COLUMN_NAME=\047POS\047 and"                \
           " SCOPE_NAME=\047"CDATA"\047;";
     print "alter table "VECTOR_TABLE_NAME                     \
           " rename column POS to DATA_ID;";
     print "update POOL_OR_MAPPING_COLUMNS"                    \
           " set   COLUMN_NAME=\047IOV_VALUE_ID\047"           \
           " where COLUMN_NAME=\047ID_ID\047 and"              \
           " SCOPE_NAME=\047"CLASS"\047;";
     print "update POOL_OR_MAPPING_COLUMNS"                    \
           " set   COLUMN_NAME=\047IOV_VALUE_ID\047"           \
           " where COLUMN_NAME=\047ID_ID\047 and"              \
           " SCOPE_NAME=\047"CDATA"\047;";
     print "alter table "VECTOR_TABLE_NAME                     \
           " rename column ID_ID to IOV_VALUE_ID;";
     print "update POOL_OR_MAPPING_ELEMENTS"                   \
           " set TABLE_NAME=\047DT_T0CELL\047"                 \
           " where TABLE_NAME=\047"VECTOR_TABLE_NAME"\047;";
     print "alter table "VECTOR_TABLE_NAME                     \
           " rename to DT_T0CELL;";
     print "update POOL_OR_MAPPING_COLUMNS"                    \
           " set COLUMN_NAME=\047IOV_VALUE_ID\047"             \
           " where COLUMN_NAME=\047ID\047 and"                 \
           " VARIABLE_NAME=\047"CLASS"\047;";
     print "alter table "OBJECT_TABLE_NAME                     \
           " rename column ID to IOV_VALUE_ID;";
     print "alter table "OBJECT_TABLE_NAME                     \
           " add ( TIME NUMBER(38) default 0 );";
     print "update POOL_OR_MAPPING_COLUMNS"                    \
           " set COLUMN_NAME=\047DATA_VERSION\047"             \
           " where COLUMN_NAME=\047DATAVERSION\047 and"        \
           " SCOPE_NAME=\047"CLASS"\047;";
     print "alter table "OBJECT_TABLE_NAME                     \
           " rename column DATAVERSION to DATA_VERSION;";
     print "alter table "OBJECT_TABLE_NAME"  rename to "       \
                         OBJECT_TABLE_NAME"X;";
     print "alter table "OBJECT_TABLE_NAME"X rename to "       \
                        "DTT0;"}
