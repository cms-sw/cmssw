#!/bin/tcsh

foreach db(`ls DBs | sed "s#.db##"`)
    echo "Setting up " $db
    rm -f ship/ship$db.csh
    rm -f ship/"$db"_T.txt
    rm -f ship/"$db"_WP.txt
    cat templates/shipDB.csh | sed "s#NAME#$db#g" > ship/ship$db.csh
    cat templates/templateForDropbox.txt | sed "s#NAME#$db#g" | sed "s#TYPE#table#g" | sed "s#TSHORT#T#g" > ship/"$db"_T.txt
    cat templates/templateForDropbox.txt | sed "s#NAME#$db#g" | sed "s#TYPE#wp#g" | sed "s#TSHORT#WP#g" > ship/"$db"_WP.txt
end
