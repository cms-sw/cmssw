#! /bin/bash -e
#
# Run `edmTypeInfo` and compare its output against the expected results.

function can_run() {
  local CMD="$1"

  OUTPUT=$(${CMD})
}

function compare() {
  local CMD="$1"
  local EXPECTED="$2"
  local OUTPUT
  OUTPUT=$(${CMD})

  if [ "${OUTPUT}" != "${EXPECTED}" ]; then
    echo "Error: unexpected output"
    echo "Command:"
    echo "${CMD}"
    echo
    echo "Expected output:"
    echo "${EXPECTED}"
    echo
    echo "Actual output:"
    echo "${OUTPUT}"
    exit 1
  fi

  return 0
}


# Run edmTypeInfo -h
CMD="edmTypeInfo -h"
can_run "${CMD}"

# Run edmTypeInfo -v
CMD="edmTypeInfo -v"
can_run "${CMD}"

# Run edmTypeInfo edmtest::reflection::IntObject
CMD="edmTypeInfo edmtest::reflection::IntObject"
EXPECTED='`edmtest::reflection::IntObject` resolves to `edmtest::reflection::IntObject`
  with friendly class name `edmtestreflectionIntObject`
  with type info `N7edmtest10reflection9IntObjectE`'
compare "${CMD}" "${EXPECTED}"
