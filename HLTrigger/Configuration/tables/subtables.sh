#! /bin/bash
#
# utility functions used to generate HLT tables from master table in ConfDB
#

function getPathList() {
  local DATA=$(edmConfigFromDB --cff --configName $MASTER --noedsources --noes --noservices --nosequences --nomodules)
  if echo "$DATA" | grep -q 'Exhausted Resultset\|CONFIG_NOT_FOUND'; then
    echo "Error: $MASTER is not a valid HLT menu"
    exit 1
  fi
  echo "$DATA" | sed -ne's/ *= *cms.\(End\)\?Path.*//p'
}

function makeCreateConfig() {
  [ -d $CMSSW_BASE/src/EventFilter/ConfigDB ]                                            || addpkg EventFilter/ConfigDB V01-05-08
  [ -f $CMSSW_BASE/src/EventFilter/ConfigDB/classes/confdb/db/ConfDBCreateConfig.class ] || ant -f $CMSSW_BASE/src/EventFilter/ConfigDB/build.xml
}

function loadConfiguration() {
  case "$1" in 
    hltdev)
      # hltdev
      DBHOST="cmsr1-v.cern.ch"
      DBNAME="cms_cond.cern.ch"
      DBUSER="cms_hltdev_writer"
      PWHASH="7a901914acb45efc107723c6d15c1bbf"
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

function createSubtables() {
  local DATABASE="$1"; shift
  local MASTER="$1"; shift
  local TARGET="$1"; shift
  local TABLES="$@"

  # load defaults for the selected database
  loadConfiguration "$DATABASE"

  # dump the requested configuration
  echo "ConfDB master: $DATABASE:$MASTER"
  echo "Subtables:     $TABLES"
  echo "Created under: $DATABASE:$TARGET/"

  # expand the wildcards in the path names in each subtables
  local LIST=$(getPathList $MASTER)
  local FAIL=0
  for TABLE in $TABLES; do
    echo "Parsing table $TABLE ..."
    rm -f $TABLE.paths
    cat "$TABLE.txt" | while read LINE; do
      PATTERN=$(echo $LINE | sed -e's/ *#.*//' -e's/^/\\</' -e's/$/\\>/' -e's/?/./g' -e's/\*/.*/g')
      [ "$PATTERN" == "\<\>" ] && continue
      echo "$LIST" | grep "$PATTERN" >> "$TABLE.paths"
      if (( $? != 0 )); then
        echo "Error: pattern \"$LINE\" does not match any paths" 1>&2
        FAIL=1
      fi
    done
  done
  if (( $FAIL )); then
    echo "Error: one or more patterns do not match any paths, exiting." 1>&2
    exit 1
  fi

  # ask for the ConfDB password, and validate its hash
  local PASSWORD=""
  read -p "Enter password for DB: " -s PASSWORD
  echo

  if [ "$(echo "$PASSWORD" | md5sum | cut -c1-32)" != "$PWHASH" ]; then
    echo "Incorrect password, exiting." 1>&2
    for TABLE in $TABLES; do
      # clean up 
      rm -f "$TABLE.paths"
    done
    exit 1
  fi

  # make sure the needed sripts are available
  makeCreateConfig

  # extract each subtable
  for TABLE in $TABLES; do
    runCreateConfig "$DATABASE" "$PASSWORD" "$MASTER" "$TABLE.paths" "$TARGET/$TABLE"
    # clean up
    rm -f "$TABLE.paths"
  done
}
