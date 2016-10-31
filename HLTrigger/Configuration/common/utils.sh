#! /bin/bash

function parse_HLT_schema() {
  # check input
  if [ $# != 1 ]; then
    echo 'parse_HLT_schema: wrong number of parameters' 1>&2
    return 1
  fi

  # default values
  Vx="v2"
  DB="$1"

  # parse the connection string [version/]database
  if [[ "$DB" =~ .*/.* ]]; then
    Vx=`echo "$DB" | cut -d/ -f1`
    DB=`echo "$DB" | cut -d/ -f2`
  fi

  echo "$Vx" "$DB"
}

function parse_HLT_menu() {
  # check input
  if [ $# != 1 ]; then
    echo 'parse_HLT_menu: wrong number of parameters' 1>&2
    return 1
  fi

  # default values
  Vx="v2"
  DB="offline"
  MENU="$1"

  # parse the connection string [[version/]database:]menu
  if [[ "$1" =~ .*:.*  ]]; then
    MENU=`echo "$1" | cut -d: -f2`
    DB=`echo "$1" | cut -d: -f1`
    if [[ "$DB" =~ .*/.* ]]; then
      Vx=`echo "$DB" | cut -d/ -f1`
      DB=`echo "$DB" | cut -d/ -f2`
    fi
  fi

  echo "$Vx" "$DB" "$MENU"
}
