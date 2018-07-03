#!/bin/bash -ex
cdump_dir=${CMSSW_BASE}
for dir in ${CMSSW_BASE} ${CMSSW_RELEASE_BASE} ${CMSSW_FULL_RELEASE_BASE} ; do
  if [ -e ${dir}/test/${SCRAM_ARCH}/cdump ] ; then
    cdump_dir=${dir}
    break
  fi
done
rm -f Alignment_Geners_cdump.out
${cdump_dir}/test/${SCRAM_ARCH}/cdump -f ${cdump_dir}/src/Alignment/Geners/test/archive.gsbmf > Alignment_Geners_cdump.out 2>&1
if [ $(diff Alignment_Geners_cdump.out ${cdump_dir}/src/Alignment/Geners/test/cdump.ref | wc -l) -gt 0 ] ;  then
    echo "!!!! Catalog dump regression test FAILED"
else
    echo "**** Catalog dump regression test passed"
fi
rm -f Alignment_Geners_cdump.out
