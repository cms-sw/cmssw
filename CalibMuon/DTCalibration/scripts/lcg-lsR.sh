#!/bin/bash
# recursive lcg-ls

help()
{
  echo
  echo "$0 performs a recursive lcg-ls on the SE given by its SURL: srm://hostname[:port]/path"
  echo
  echo "Usage:"
  echo "$0 [-h|--help]"
  echo "$0 [-f | --only-files ]"
  echo
  echo "  --f | --only-files: only list the files, hide the directories."
  echo
  echo "  -h, --help: display this help"
  echo
  exit 1
}

while [ ! -z "$1" ]
do
  case "$1" in
    -f | --only-files ) ONLYFILES="true";;
    -h | --help ) help;;
    *) SURL=$1;;
  esac
  shift
done

if test -z "{SURL}"; then
  help
  exit
fi

LCGLS_OPTS="--connect-timeout 30 --sendreceive-timeout 900 --bdii-timeout 30 --srm-timeout 300"

# Recursive ls function
function recls {
  local S=$1

  # Get the file rights (ie.g. drwxr-x---)
  prot=`lcg-ls -ld $LCGLS_OPTS "$S" | head -1 | cut -d ' ' -f1`
  difi=${prot:0:1}

  # Is this a directory?
  if [ "$difi" == "d" ]; then
    if test -z "$ONLYFILES"; then
      lcg-ls -l $LCGLS_OPTS $S
    else
      lcg-ls -l $LCGLS_OPTS $S | egrep -v "^d"
    fi

    flist=`lcg-ls $LCGLS_OPTS $S`
    for f in $flist; do
      file=$S/`basename $f`
      recls $file
    done
  fi
}

recls $SURL

