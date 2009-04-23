#!/bin/csh

setenv CELL_MAP_VERSION $1
setenv  ROB_MAP_VERSION $2

setenv DB_SYS oracle://cms_val_lb.cern.ch/CMS_VAL_DT_POOL_OWNER
setenv POOL_AUTH_USER CMS_VAL_DT_POOL_WRITER
setenv POOL_AUTH_PASSWORD val_dt_wri_1031

setenv TMPFILE /tmp/iovtoken`date +%s`

cmscond_build_iov -b -c ${DB_SYS} -t DTREADOUTMAPPING -n DTReadOutMapping -d libCondFormatsDTObjectsCapabilities -T ${CELL_MAP_VERSION}"_"${ROB_MAP_VERSION}

#cmscond_build_iov -b -c ${DB_SYS} -t DTREADOUTMAPPING -n DTReadOutMapping -s 6E1247AF-2A00-DA11-96EE-000E0C4DE431 > ${TMPFILE}

#setenv IOV_TOKEN `tail -1 ${TMPFILE}`

#cmscond_build_metadata -c ${DB_SYS} -i "${IOV_TOKEN}" -t ${CELL_MAP_VERSION}"_"${ROB_MAP_VERSION}

#rm -f ${TMPFILE}

