#!/bin/bash
ME="`basename $0`"
if [ -z "$FRONTIER_CLIENT" ]; then
    echo "$ME: error, \$FRONTIER_CLIENT isn't set"
    exit 1
fi
exec ${FRONTIER_CLIENT%/}/bin/$ME "$@"
