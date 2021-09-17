#!/bin/bash -ex
if [ $(klist | grep 'Default principal' | grep cmsbuild | wc -l) -eq 0 ] ; then
  echo "Only run for cmsbuild user which has the rights to copy LHE files"
  exit 0
fi
CMSEOS_BASE="/eos/cms/store/user/cmsbuild/unittest/lhe"
export CMSEOS_LHE_ROOT_DIRECTORY="${CMSEOS_BASE}/ref"
LHEtoEOSManager=${CMSSW_BASE}/src/GeneratorInterface/LHEInterface/scripts/cmsLHEtoEOSManager.py
REF_FILE=$(${LHEtoEOSManager} -l 1 | grep 'lhe.xz$' | tail -1)
if [ $REF_FILE = "" ] ; then
  echo "ERROR: Unable to find reference LHE file"
  exit 1
fi
ERR=0
rm -rf test_cmsLHEtoEOSManager ; mkdir -p test_cmsLHEtoEOSManager
pushd test_cmsLHEtoEOSManager
  xrdcp root://eoscms.cern.ch/${CMSEOS_LHE_ROOT_DIRECTORY}/1/${REF_FILE} ${REF_FILE}
  xz -d ${REF_FILE}
  REF_FILE=$(echo $REF_FILE | sed 's|.xz$||')
  UNQ_NAME=$(date +%s)-$(echo $(hostname) $$ | md5sum | sed 's| .*||').lhe
  mv ${REF_FILE} ${UNQ_NAME}
  export CMSEOS_LHE_ROOT_DIRECTORY="${CMSEOS_BASE}/ibs"
  if ${LHEtoEOSManager} -u 1 --compress -f ${UNQ_NAME} ; then
    ${LHEtoEOSManager} --force -u 1 -f ${UNQ_NAME}.xz || ERR=1
  else
    ERR=1
  fi
popd
rm -rf test_cmsLHEtoEOSManager
xrdfs root://eoscms.cern.ch/ rm    ${CMSEOS_LHE_ROOT_DIRECTORY}/1/${UNQ_NAME}.xz || true
exit $ERR
