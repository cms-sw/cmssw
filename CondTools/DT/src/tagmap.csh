#!/bin/csh

setenv CELL_MAP_VERSION $1
setenv  ROB_MAP_VERSION $2

setenv DB_SYS sqlite_file:testfile.db
setenv POOL_AUTH_USER     user
setenv POOL_AUTH_PASSWORD pass

setenv TMPFILE /tmp/iovtoken`date +%s`

cmscond_build_iov -b -c ${DB_SYS} -t DTREADOUTMAPPING -n DTReadOutMapping -d libCondFormatsDTObjectsCapabilities -T ${CELL_MAP_VERSION}"_"${ROB_MAP_VERSION}

#cmscond_build_iov -b -c ${DB_SYS} -t DTREADOUTMAPPING -n DTReadOutMapping -s 6E1247AF-2A00-DA11-96EE-000E0C4DE431 > ${TMPFILE}

#setenv IOV_TOKEN `tail -1 ${TMPFILE}`

#cmscond_build_metadata -c ${DB_SYS} -i "${IOV_TOKEN}" -t ${CELL_MAP_VERSION}"_"${ROB_MAP_VERSION}

#rm -f ${TMPFILE}

