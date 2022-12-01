#!/bin/bash
#
# utility functions used to generate HLT tables from master table in ConfDB
#

# db-proxy configuration
DBPROXY=""
DBPROXYHOST="localhost"
DBPROXYPORT="8080"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dbproxy) DBPROXY="--dbproxy"; shift;;
    --dbproxyhost) DBPROXYHOST="$2"; shift; shift;;
    --dbproxyport) DBPROXYPORT="$2"; shift; shift;;
    *) shift;;
  esac
done

# load common HLT functions
if [ -f "$CMSSW_BASE/src/HLTrigger/Configuration/common/utils.sh" ]; then
  source "$CMSSW_BASE/src/HLTrigger/Configuration/common/utils.sh"
elif [ -f "$CMSSW_RELEASE_BASE/src/HLTrigger/Configuration/common/utils.sh" ]; then
  source "$CMSSW_RELEASE_BASE/src/HLTrigger/Configuration/common/utils.sh"
else
  exit 1
fi

CONFDB_TAG="HEAD"

# if set, remove the ConfDB working directory
private=false

function cleanup() {
  local TABLES="$@"

  # clean up
  for TABLE in $TABLES; do
    rm -f "${TABLE}_expanded.txt"
  done

  if $private; then
    rm -rf $workDir
  fi
}

function getPathList() {
  [ "x${DBPROXY}" = "x" ] || local DBPROXYOPTS="${DBPROXY} --dbproxyhost ${DBPROXYHOST} --dbproxyport ${DBPROXYPORT}"
  local DATA=$(hltConfigFromDB --${Vx} --${DB} --cff --configName ${MASTER} \
    --noedsources --noes --noservices --nosequences --nomodules ${DBPROXYOPTS})
  if echo "${DATA}" | grep -q 'Exhausted Resultset\|CONFIG_NOT_FOUND'; then
    echo "Error: $MASTER is not a valid HLT menu"
    exit 1
  fi
  echo "${DATA}" | sed -ne's/ *= *cms.\(Final\|End\)\?Path.*//p'
}

function checkJars() {
  local BASE="$1"; shift
  local JARS="$@"
  for F in "$BASE/$JARS"; do
    [ -f "$F" ] || return 1
  done
  return 0
}

function makeCreateConfig() {
  local baseDir="/afs/cern.ch/user/c/confdb/www/${Vx}/lib"
  local baseUrl="http://confdb.web.cern.ch/confdb/${Vx}/lib"
  local JARS="ojdbc8.jar cmssw-evf-confdb-gui.jar"
  workDir="$baseDir"

  # try to read the .jar files from AFS, or download them
  if checkJars "$baseDir" $JARS; then
    # read the .jar fles from AFS
    workDir="$baseDir"
  else
    # try to use $CMSSW_BASE/tmp
    mkdir -p "$CMSSW_BASE/tmp/confdb"
    if [ -d "$CMSSW_BASE/tmp/confdb" ]; then
      workDir="$CMSSW_BASE/tmp/confdb"
    else
      workDir=$(mktemp -d confdb.XXXXXXXXXX)
      private=true
    fi
    # download the .jar files
    for JAR in $JARS; do
      # check if the file is already present
      if [ -f $workDir/$JAR ]; then
        continue
      fi
      # download to a temporary file and use an atomic move (in case another instance is downloading the same file)
      local TMPJAR=$(mktemp -p "$workDir" .${JAR}.XXXXXXXXXX)
      curl -s -L "$baseUrl/$JAR" -o "$TMPJAR"
      mv -n "$TMPJAR" "$workDir/$JAR"
      rm -f "$TMPJAR"
    done
  fi

  CLASSPATH=
  for JAR in $JARS; do
    CLASSPATH="$CLASSPATH${CLASSPATH:+:}$workDir/$JAR"
  done
}

function loadConfiguration() {
  case "$1" in
    # v1 offline aka "hltdev"
    "v1/offline" | "v1/hltdev")
      DBHOST="cmsr1-s.cern.ch,cmsr2-s.cern.ch,cmsr3-s.cern.ch"
      [ "x${DBPROXY}" = "x" ] || DBHOST="10.116.96.89,10.116.96.139,10.116.96.105"
      DBNAME="cms_cond.cern.ch"
      DBUSER="cms_hltdev_writer"
      PWHASH="0196d34dd35b04c0f3597dc89fbbe6e2"
      ;;
    # v2 offline
    "v2/offline")
      DBHOST="cmsr1-s.cern.ch,cmsr2-s.cern.ch,cmsr3-s.cern.ch"
      [ "x${DBPROXY}" = "x" ] || DBHOST="10.116.96.89,10.116.96.139,10.116.96.105"
      DBNAME="cms_cond.cern.ch"
      DBUSER="cms_hlt_gdr_w"
      PWHASH="0196d34dd35b04c0f3597dc89fbbe6e2"
      ;;
    # converter=v3*, db=run3
    "v3/run3" | "v3-beta/run3" | "v3-test/run3")
      DBHOST="cmsr1-s.cern.ch,cmsr2-s.cern.ch,cmsr3-s.cern.ch"
      [ "x${DBPROXY}" = "x" ] || DBHOST="10.116.96.89,10.116.96.139,10.116.96.105"
      DBNAME="cms_hlt.cern.ch"
      DBUSER="cms_hlt_v3_w"
      PWHASH="0196d34dd35b04c0f3597dc89fbbe6e2"
      ;;
    # converter=v3*, db=dev
    "v3/dev" | "v3-beta/dev" | "v3-test/dev")
      DBHOST="cmsr1-s.cern.ch,cmsr2-s.cern.ch,cmsr3-s.cern.ch"
      [ "x${DBPROXY}" = "x" ] || DBHOST="10.116.96.89,10.116.96.139,10.116.96.105"
      DBNAME="cms_hlt.cern.ch"
      DBUSER="cms_hlt_gdrdev_w"
      PWHASH="0196d34dd35b04c0f3597dc89fbbe6e2"
      ;;
    *)
      # see https://github.com/fwyzard/hlt-confdb/blob/confdbv2/test/runCreateConfig
      echo "Error, unknown database \"$1\", exiting."
      exit 1
      ;;
  esac
}

function runCreateConfig() {
  [ "x${DBPROXY}" = "x" ] || local DBPROXYOPTS="-DsocksProxyHost=${DBPROXYHOST} -DsocksProxyPort=${DBPROXYPORT}"
  loadConfiguration "$1"
  java \
    -Djava.security.egd=file:///dev/urandom \
    -Doracle.jdbc.timezoneAsRegion=false \
    ${DBPROXYOPTS} \
    -Xss32M \
    -Xmx1024m \
    -classpath "${CLASSPATH}" \
    confdb.db.ConfDBCreateConfig \
    --dbHost "${DBHOST}" \
    --dbName "${DBNAME}" \
    --dbUser "${DBUSER}" \
    --dbPwrd $2 \
    --master $3 \
    --paths $4 \
    --name $5
}

# expands the patterns in "TABLE.txt" into "TABLE_expanded.txt"
function expandSubtable() {
  local TABLE="$1"
  local LIST="$2"
  local FAIL=0

  echo "Parsing table: $TABLE ..."
  rm -f ${TABLE}_expanded.txt
  cat "$TABLE.txt" | while read LINE; do
    PATTERN=$(echo $LINE | sed -e's/ *#.*//' -e's/^/\\</' -e's/$/\\>/' -e's/?/./g' -e's/\*/.*/g')
    [ "$PATTERN" == "\<\>" ] && continue
    echo "$LIST" | grep "$PATTERN" >> "${TABLE}_expanded.txt"
    if (( $? != 0 )); then
      echo "Error: pattern \"$LINE\" does not match any paths" 1>&2
      FAIL=1
    fi
  done

  return $FAIL
}


function readPassword() {
  # ask for the ConfDB password, and validate its hash
  loadConfiguration "$DATABASE"
  PASSWORD=""
  read -p "Enter password for DB: " -s PASSWORD
  echo

  if [ "$(echo "$PASSWORD" | tr 'a-z' 'A-Z' | md5sum | cut -c1-32)" != "$PWHASH" ]; then
    echo "Incorrect password, exiting." 1>&2
    exit 1
  fi
}


function createSubtables() {
  local DATABASE="$1"; shift
  local MASTER="$1";   shift
  local TARGET="$1";   shift
  local TABLES="$@"

  # extract the schema version from the database name
  local Vx DB
  read Vx DB <<< $(parse_HLT_schema "$DATABASE")
  local DATABASE="${Vx}/${DB}"

  # dump the requested configuration
  echo "ConfDB master: $DATABASE:$MASTER"
  echo "Subtables:     $TABLES"
  echo "Created under: $DATABASE:$TARGET"

  # install a clean up hook
  trap "cleanup $TABLES; exit 1" INT TERM EXIT

  # expand the wildcards in the path names in each subtables
  local LIST=$(getPathList $MASTER)
  local FAIL=0
  for TABLE in $TABLES; do
    expandSubtable "$TABLE" "$LIST" || FAIL=1
  done
  if (( $FAIL )); then
    echo "Error: one or more patterns do not match any paths, exiting." 1>&2
    exit 1
  fi

  # ask the user for the database password
  readPassword

  # make sure the needed scripts are available
  makeCreateConfig

  # extract each subtable
  for TABLE in $TABLES; do
    runCreateConfig "$DATABASE" "$PASSWORD" "$MASTER" "${TABLE}_expanded.txt" $(echo "$TARGET" | sed -e"s|TABLE|$TABLE|")
  done

  # remove clean up hook, and call explicit cleanup
  trap - INT TERM EXIT
  cleanup $TABLES
}
