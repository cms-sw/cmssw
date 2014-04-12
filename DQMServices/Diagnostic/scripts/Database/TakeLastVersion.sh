#! /bin/bash

function TakeLastVersion ()
{
    local FileDir=$1
    local FileListName=$2
    echo "FileDir = $FileDir"
    echo "FileListName = $FileListName"

    echo "Take the last version file of each run"

    declare -a RunNumberList
    RunNumberList=(`ls ${FileDir}*.root | awk -F"/" '{print $NF}' | awk -F"_" '{print $3}' | sort | uniq`)
    #  ls ${FileDir}*.root | awk -F"/" '{print $NF}'
    #  echo "RunNumberList: $RunNumberList"

    local k=0
    local ListSize=${#RunNumberList[*]}
    echo "ListSize: $ListSize"

    rm -f $FileListName
    while [ "$k" -lt "$ListSize" ]
        do
        local Version=`ls ${FileDir}*.root | grep ${RunNumberList[$k]} | awk -F"/" '{print $NF}' | awk -F"_" '{print $2}' | sort | tail -1`
        local FileName=`ls ${FileDir}*.root | grep ${RunNumberList[$k]} | grep $Version`
        echo $FileName >> ${FileListName}
        echo "RunNumber: ${RunNumberList[$k]}  FileName: $FileName"
        let "k+=1"
    done
}
