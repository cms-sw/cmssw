#!/bin/tcsh

#Unique version number adn tag for the DB, should match makeAll.csh
set version=v9
#set tag=PerformancePayloadFromTable
set tag=PerformancePayloadFromBinnedTFormula

foreach db(`ls DBs | sed "s#.db##"`)
    echo "Setting up " $db
    rm -f ship/ship$db.csh
    rm -f ship/"$db"_T.txt
    rm -f ship/"$db"_WP.txt
    cat templates/shipDB.csh | sed "s#NAME#$db#g" > ship/ship$db.csh
    cat templates/templateForNEWDropbox.txt | sed "s#TAG#$tag#g" | sed "s#NAME#$db#g" | sed "s#TYPE#table#g" | sed "s#TSHORT#T#g"  | sed "s#VERSION#$version#g" > ship/"$db"_T.txt
    cat templates/templateForNEWDropbox_WP.txt | sed "s#NAME#$db#g" | sed "s#TYPE#wp#g" | sed "s#TSHORT#WP#g"  | sed "s#VERSION#$version#g" > ship/"$db"_WP.txt

    pwd
    cd DBs
    pwd
    cmscond_export_iov -s sqlite_file:../DBs/$db.db -d sqlite_file:"$db"_T.db -t "$db"_T
    cmscond_export_iov -s sqlite_file:../DBs/$db.db -d sqlite_file:"$db"_WP.db -t "$db"_WP
    cd ..
    mv DBs/"$db"_T.db ship
    mv DBs/"$db"_WP.db ship
    pwd

end
