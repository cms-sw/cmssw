#! /bin/sh
# CRAB related Stuff
export CRABDIR=/analysis/sw/CRAB/CRAB_1_4_2/CRAB
export CRABSCRIPT=${CRABDIR}/script

CRABPATH=${CRABDIR}/python
CRABDLSAPIPATH=${CRABDIR}/DLSAPI
export CRABPYTHON=${CRABDIR}/python
export CRABDBSAPIPYTHON=${CRABDIR}/DBSAPI
export CRABDLSAPIPYTHON=${CRABDIR}/DLSAPI
export CRABPSETPYTHON=${CRABDIR}/PsetCode

if [ -z "$PATH" ]; then
export PATH=${CRABPATH}
else
export PATH=${CRABPATH}:${PATH}
fi
if [ -z "$PYTHONPATH" ]; then
export PYTHONPATH=${CRABPYTHON}:${CRABDBSAPIPYTHON}:${CRABDLSAPIPYTHON}:${CRABPSETPYTHON}
else
export PYTHONPATH=${PYTHONPATH}:${CRABPYTHON}:${CRABDBSAPIPYTHON}:${CRABDLSAPIPYTHON}:${CRABPSETPYTHON}
fi

# BOSS related Stuff
source /analysis/sw/CRAB/CRAB_1_4_2/CRAB/Boss/BOSS_4_2_4/bossenv.sh

# check whether central boss db is configured

# check if .bossrc dir exists

echo MYHOME $MYHOME

if [ ! -d $MYHOME/.bossrc ]; then
  mkdir $MYHOME/.bossrc
fi

# check if *clad files exist

if [ ! -e $MYHOME/.bossrc/BossConfig.clad ]; then
  if [ -e $MYHOME/BossConfig.clad ]; then
    cp  $MYHOME/BossConfig.clad $MYHOME/.bossrc/BossConfig.clad
  else
    echo "User-boss DB not installed: run $\CRABDIR/configureBoss"
    return 1
  fi
fi
if [ ! -e $MYHOME/.bossrc/SQLiteConfig.clad ]; then
  if [ -e $MYHOME/SQLiteConfig.clad ]; then
    cp $MYHOME/SQLiteConfig.clad $MYHOME/.bossrc/SQLiteConfig.clad
  else
    echo "User-boss DB not installed: run $\CRABDIR/configureBoss"
    return 1
  fi
fi
if [ ! -e $MYHOME/.bossrc/MySQLRTConfig.clad ]; then
  if [ -e $MYHOME/MySQLRTConfig.clad ]; then
    cp  $MYHOME/MySQLRTConfig.clad  $MYHOME/.bossrc/MySQLRTConfig.clad
  else
    echo "User-boss DB not installed: run $\CRABDIR/configureBoss"
    return 1
  fi
fi
# now check a boss command to see if boss DB is up and running
boss clientID 1>test.txt
if [ `boss clientID 1>/dev/null | grep -c "not correctly configured"` -ne 0 ]; then
echo inside
  echo "User-boss DB not installed: run $\CRABDIR/configureBoss"
    return 1
fi
