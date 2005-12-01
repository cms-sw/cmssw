BEGIN {CLASS="DTReadOutMapping"; \
       CDATA="DTReadOutMapping::readOutChannelDriftTubeMap"; \
       SCOPE="DTReadOutMapping::readOutChannelDriftTubeMap::DTReadOutGeometryLink"}
($2==     "::" && $3=="DTReadOutMapping") \
                                   {OBJECT_TABLE_NAME=$5}
($2==SCOPE"::" && $3==   "cellId") {COLUMN_NAME="CELL"; VECTOR_TABLE_NAME=$5};
($2==SCOPE"::" && $3==  "layerId") {COLUMN_NAME="LAYER"};
($2==SCOPE"::" && $3==     "slId") {COLUMN_NAME="SUPERLAYER"};
($2==SCOPE"::" && $3=="stationId") {COLUMN_NAME="STATION"};
($2==SCOPE"::" && $3== "sectorId") {COLUMN_NAME="SECTOR"};
($2==SCOPE"::" && $3==  "wheelId") {COLUMN_NAME="WHEEL"};
($2==SCOPE"::" && $3==    "dduId") {COLUMN_NAME="DDU"};
($2==SCOPE"::" && $3==    "rosId") {COLUMN_NAME="ROS"};
($2==SCOPE"::" && $3==    "robId") {COLUMN_NAME="ROB"};
($2==SCOPE"::" && $3==    "tdcId") {COLUMN_NAME="TDC"};
($2==SCOPE"::" && $3=="channelId") {COLUMN_NAME="CHANNEL"};
($2==SCOPE"::") {
     print "update POOL_OR_MAPPING_COLUMNS"                    \
           " set COLUMN_NAME=\047"                             \
                 COLUMN_NAME"\047"                             \
           " where VARIABLE_NAME=\047"$3"\047 and"             \
           " SCOPE_NAME=\047"SCOPE"\047;";
     print "alter table "$5                                    \
           " rename column "$4" to "COLUMN_NAME";"}
END {print "update POOL_OR_MAPPING_COLUMNS"                    \
           " set   COLUMN_NAME=\047CONNECTION_ID\047"          \
           " where COLUMN_NAME=\047POS\047 and"                \
           " SCOPE_NAME=\047"CLASS"\047;";
     print "update POOL_OR_MAPPING_COLUMNS"                    \
           " set   COLUMN_NAME=\047CONNECTION_ID\047"          \
           " where COLUMN_NAME=\047POS\047 and"                \
           " SCOPE_NAME=\047"CDATA"\047;";
     print "alter table "VECTOR_TABLE_NAME                     \
           " rename column POS to CONNECTION_ID;";
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
           " set TABLE_NAME=\047DTREADOUTCONNECTION\047"       \
           " where TABLE_NAME=\047"VECTOR_TABLE_NAME"\047;";
     print "alter table "VECTOR_TABLE_NAME                     \
           " rename to DTREADOUTCONNECTION;";
     print "update POOL_OR_MAPPING_COLUMNS"                    \
           " set COLUMN_NAME=\047IOV_VALUE_ID\047"             \
           " where COLUMN_NAME=\047ID\047 and"                 \
           " VARIABLE_NAME=\047"CLASS"\047;";
     print "alter table "OBJECT_TABLE_NAME                     \
           " rename column ID to IOV_VALUE_ID;";
     print "update POOL_OR_MAPPING_COLUMNS"                    \
           " set COLUMN_NAME=\047CELL_MAP_VERSION\047"         \
           " where COLUMN_NAME=\047CELLMAPVERSION\047 and"     \
           " SCOPE_NAME=\047"CLASS"\047;";
     print "alter table "OBJECT_TABLE_NAME                     \
           " rename column CELLMAPVERSION to CELL_MAP_VERSION;";
     print "update POOL_OR_MAPPING_COLUMNS"                    \
           " set COLUMN_NAME=\047ROB_MAP_VERSION\047"          \
           " where COLUMN_NAME=\047ROBMAPVERSION\047 and"      \
           " SCOPE_NAME=\047"CLASS"\047;";
     print "alter table "OBJECT_TABLE_NAME                     \
           " rename column ROBMAPVERSION to ROB_MAP_VERSION;";
     print "alter table "OBJECT_TABLE_NAME"  rename to "       \
                         OBJECT_TABLE_NAME"X;";
     print "alter table "OBJECT_TABLE_NAME"X rename to "       \
                        "DTREADOUTMAPPING;"}
