#! /bin/bash

function parse_HLT_menu() {
  # check input
  if [ $# != 1 ]; then
    echo 'parse_HLT_menu: wrong number of parameters' 1>&2
    return 1
  fi

  # default values
  DB="offline"
  Vx="v2"
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
