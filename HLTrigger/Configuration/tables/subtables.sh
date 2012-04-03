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
  local DATA=$(edmConfigFromDB --cff --configName $MASTER --noedsources --noes --noservices --nosequences --nomodules)
  if echo "$DATA" | grep -q 'Exhausted Resultset\|CONFIG_NOT_FOUND'; then
    echo "Error: $MASTER is not a valid HLT menu"
    exit 1
  fi
  echo "$DATA" | sed -ne's/ *= *cms.\(End\)\?Path.*//p'
}

function makeCreateConfig() {
  [ -d $CMSSW_BASE/src/EventFilter/ConfigDB ]                                            || addpkg EventFilter/ConfigDB $CONFDB_TAG
  [ -f $CMSSW_BASE/src/EventFilter/ConfigDB/classes/confdb/db/ConfDBCreateConfig.class ] || ant -f $CMSSW_BASE/src/EventFilter/ConfigDB/build.xml
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
      # see $CMSSW_BASE/src/EventFilter/ConfigDB/test/runCreateConfig for other possible settings
      echo "Error, unnown database \"$1\", exiting."
      exit 1
      ;;
  esac
}

function runCreateConfig() {
  loadConfiguration "$1"
  java \
    -Xmx1024m \
    -classpath "$CMSSW_BASE/src/EventFilter/ConfigDB/ext/ojdbc14.jar:$CMSSW_BASE/src/EventFilter/ConfigDB/lib/cmssw-evf-confdb-gui.jar" \
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
