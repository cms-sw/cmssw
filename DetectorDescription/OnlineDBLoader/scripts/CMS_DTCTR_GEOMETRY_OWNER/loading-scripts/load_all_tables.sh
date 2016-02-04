#!/bin/sh
# This script load all the tables in DGD using sqlldr.
# Yuyi Guo May 13, 2005 
#

echo -n 'database TNS name: '
read DB_NAME
echo -n 'schema writer name: '
read USER_NAME
echo -n 'schema writer password: '
read PASSWD
CONN_ID=$USER_NAME@$DB_NAME/$PASSWD
#echo "$CONN_ID" 

echo 'Loading Table SOLIDS' 
sqlldr $CONN_ID control=solids.ctl log=solids.log

#echo 'Loading Table boxes'
sqlldr $CONN_ID control=boxes.ctl log=boxes.log

#echo 'Loading Table trapezoids '
sqlldr $CONN_ID control=trapezoids.ctl log=trapezoids.log

#echo 'Loading Table trapezoids '
sqlldr $CONN_ID control=pseudotraps.ctl log=pseduotraps.log

#echo 'Loading Table cones '
sqlldr $CONN_ID control=cones.ctl log=cones.log

#echo 'Loading Table  polyhedras'
sqlldr $CONN_ID control=polyhedras.ctl log=polyhedras.log

#echo 'Loading Table polycones'
sqlldr $CONN_ID control=polycones.ctl log=polycones.log

#echo 'Loading Table tubesections'
sqlldr $CONN_ID control=tubesections.ctl log=tubesections.log

#echo 'Loading Table trd1s '
sqlldr $CONN_ID control=trd1s.ctl log=trd1s.log

#echo 'Loading Table rotations '
sqlldr $CONN_ID control=rotations.ctl log=rotations.log

#echo 'Loading Table booleansolids '
sqlldr $CONN_ID control=booleansolids.ctl log=booleansolids.log

#echo 'Loading Table zsections'
sqlldr $CONN_ID control=zsections.ctl log=zsections.log

#echo 'Loading Table materials'
sqlldr $CONN_ID control=materials.ctl log=materials.log

#echo 'Loading Table  compositematerials '
sqlldr $CONN_ID control=compositematerials.ctl log=compositematerials.log

#echo 'Loading Table elementarymaterials '
sqlldr $CONN_ID control=elementarymaterials.ctl log=elementarymaterials.log

#echo 'Loading Table materialfractions'
sqlldr $CONN_ID control=materialfractions.ctl log=materialfractions.log

#echo 'Loading Table categories '
sqlldr $CONN_ID control=categories.ctl log=categories.log

#echo 'Loading Table logicalparttypes & detectorparts'
sqlldr $CONN_ID control=logicalparts.ctl log=logicalparts.log
sqlldr $CONN_ID control=detectorparts.ctl log=detectorparts.log

#echo 'Loading Table  pospartsgraph & replacements '
sqlldr $CONN_ID control=pospartsgraph.ctl log= pospartsgraph.log

#echo 'Loading Table  physicalpartstree '
sqlldr $CONN_ID control=physicalpartstree.ctl log=physicalpartstree.log

#echo 'Loading Table nominalplacements '
sqlldr $CONN_ID control=nominalplacements.ctl log=nominalplacements.log
