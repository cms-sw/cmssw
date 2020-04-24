#!/bin/bash
#
# Script to create sqlite files with single-IOV tags from files with
# multi-IOV tags.
# The multi-IOV tags are kept in the resulting 'split' db file.
#
################################################################################

if [ x${1} = 'x-h' ] || [ x${1} = 'x--help' ] || [ ${#} -ne 2 ]
then
    echo "Usage: ${0} <input db name> <output db name>"
    exit 1
elif [ x${CMSSW_BASE} = 'x' ]
then
    echo "Please source a CMSSW environment."
    exit 1
fi

cp ${1} ${2}

tags=$(sqlite3 ${1} "SELECT NAME FROM TAG;")

count_copied_iovs=0
for tag in ${tags}
do
    tag_info_query="SELECT SINCE,PAYLOAD_HASH FROM IOV
                    WHERE TAG_NAME IS '${tag}';"
    tag_infos=$(sqlite3 ${1} "${tag_info_query}" | sort)
    if [ $(echo "${tag_infos}" | wc -l) -eq 1 ] # Is already a single-IOV tag?
    then
        continue
    fi

    count=0
    for tag_info in ${tag_infos}
    do
        since=$(echo ${tag_info} | cut -d'|' -f1)
        payload_hash=$(echo ${tag_info} | cut -d'|' -f2)
        new_tag=${tag}_${count}
        count=$(( count + 1 ))

        exists_query="SELECT EXISTS (
                        SELECT * FROM IOV
                        WHERE TAG_NAME IS '${new_tag}'
                        AND SINCE IS 1
                        AND PAYLOAD_HASH IS '${payload_hash}');"
        if [ $(sqlite3 ${1} "${exists_query}") -ne 0 ]
        then
            continue
        fi

        conddb_copy_iov -c sqlite_file:${2} -s ${since} -d 1 -t ${new_tag} -i ${tag}
        count_copied_iovs=$(( count_copied_iovs + 1 ))
    done
done

if [ ${count_copied_iovs} -eq 0 ]
then
    echo "   >>> '${1}' is already a 'split' db file or contains only single-IOV tags."
    echo "   >>> The created db file '${2}' is therefore just a copy of '${1}'."
fi
