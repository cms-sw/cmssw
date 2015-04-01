#! /bin/bash
#
# utility functions used to generate HLT tables from master table in ConfDB
#

CONFDB_TAG="HEAD"

function cleanup() {
  local TABLES="$@"

  # clean up
  for TABLE in $TABLES; do
    rm -f "${TABLE}_expanded.txt"
  done
}

function getPathList() {
  local DATA=$(hltConfigFromDB --cff --configName $MASTER --noedsources --noes --noservices --nosequences --nomodules)
  if echo "$DATA" | grep -q 'Exhausted Resultset\|CONFIG_NOT_FOUND'; then
    echo "Error: $MASTER is not a valid HLT menu"
    exit 1
  fi
  echo "$DATA" | sed -ne's/ *= *cms.\(End\)\?Path.*//p'
}

function makeCreateConfig() {
  # if not already present, check out and build the ConfDB converter
  if ! [ -d "$CMSSW_BASE/hlt-confdb/.git" ]; then
    mkdir -p "$CMSSW_BASE/hlt-confdb"
    git clone "https://github.com/cms-sw/hlt-confdb.git" "$CMSSW_BASE/hlt-confdb" 1>&2
  fi
  if ! [ -f "$CMSSW_BASE/hlt-confdb/lib/cmssw-evf-confdb-gui.jar" ]; then
    ant -f "$CMSSW_BASE/hlt-confdb/build.xml" gui 1>&2
  fi
}

function loadConfiguration() {
  case "$1" in 
    hltdev)
      # hltdev
      DBHOST="cmsr1-v.cern.ch"
      DBNAME="cms_cond.cern.ch"
      DBUSER="cms_hltdev_writer"
      PWHASH="0196d34dd35b04c0f3597dc89fbbe6e2"
      ;;
    *)
      # see $CMSSW_BASE/hlt-confdb/test/runCreateConfig for other possible settings
      echo "Error, unnown database \"$1\", exiting."
      exit 1
      ;;
  esac
}

function runCreateConfig() {
  loadConfiguration "$1"
  java \
    -Djava.security.egd=file:///dev/urandom \
    -Xmx1024m \
    -classpath "$CMSSW_BASE/hlt-confdb/ext/ojdbc6.jar:$CMSSW_BASE/hlt-confdb/lib/cmssw-evf-confdb-gui.jar" \
    confdb.db.ConfDBCreateConfig \
    --dbHost $DBHOST \
    --dbName $DBNAME \
    --dbUser $DBUSER \
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
  readPassword $DATABASE

  # make sure the needed sripts are available
  makeCreateConfig

  # extract each subtable
  for TABLE in $TABLES; do
    runCreateConfig "$DATABASE" "$PASSWORD" "$MASTER" "${TABLE}_expanded.txt" $(echo "$TARGET" | sed -e"s|TABLE|$TABLE|")
  done

  # remove clean up hook, and call explicit cleanup
  trap - INT TERM EXIT
  cleanup $TABLES
}
