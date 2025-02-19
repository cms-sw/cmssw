#! /bin/bash
source /afs/cern.ch/cms/caf/setup.sh


INDIR=/castor/cern.ch/cms/store/caf/user/bonato/Collisions2010/Run2010B/Oct2010/
OUTDIR=/castor/cern.ch/cms/store/caf/user/bonato/Collisions2010/Run2010A-v2/

NMOVED=0
for file in $( nsls ${INDIR} )
do
nsrename -f ${INDIR}/${file}  ${OUTDIR}/${file}
let "NMOVED=NMOVED+1"
done

echo "Moved $NMOVED files"
echo ""
echo "List of files remaining in INDIR=${INDIR}"
nsls $INDIR
