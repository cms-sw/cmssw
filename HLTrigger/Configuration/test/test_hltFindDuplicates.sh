#!/bin/bash

# Pass in name and status
function die {
  printf "\n%s: status %s\n" "$1" "$2"
  if [ $# -gt 2 ]; then
    printf "%s\n" "=== Log File =========="
    cat $3
    printf "%s\n" "=== End of Log File ==="
  fi
  exit $2
}

if [ -z "${SCRAM_TEST_PATH}" ]; then
  printf "\n%s\n\n" "ERROR -- environment variable SCRAM_TEST_PATH not defined"
  exit 1
fi

###
### test #1: "mode == 0"
###
rm -rf test_hltFindDuplicates_mode0_output

hltFindDuplicates "${SCRAM_TEST_PATH}"/test_hltFindDuplicates_cfg.py -x="--mode=0" -v 2 \
  -o test_hltFindDuplicates_mode0_output &> test_hltFindDuplicates_mode0_log \
  || die 'Failure running hltFindDuplicates (mode: 0)' $? test_hltFindDuplicates_mode0_log

cat <<@EOF > test_hltFindDuplicates_mode0_groups_expected
# A3 (d3x)
d3x
d3y
m3x
m3y

# F2 (d2x)
d2x
d2y
m2x
m2y

# P1 (d1x)
d1x
d1y
m1x
m1y
@EOF

diff test_hltFindDuplicates_mode0_groups_expected test_hltFindDuplicates_mode0_output/groups.txt \
  || die "Unexpected differences in groups.txt output of hltFindDuplicates (mode: 0)" $?

###
### test #2: "mode == 1"
###
rm -rf test_hltFindDuplicates_mode1_output

hltFindDuplicates "${SCRAM_TEST_PATH}"/test_hltFindDuplicates_cfg.py -x="--mode=1" -v 2 \
  -o test_hltFindDuplicates_mode1_output &> test_hltFindDuplicates_mode1_log \
  || die 'Failure running hltFindDuplicates (mode: 1)' $? test_hltFindDuplicates_mode1_log

cat <<@EOF > test_hltFindDuplicates_mode1_groups_expected
# A3 (d3x)
d3x
d3y
m3x

# F2 (d2x)
d2x
d2y
m2x

# P1 (d1x)
d1x
d1y
m1x
@EOF

diff test_hltFindDuplicates_mode1_groups_expected test_hltFindDuplicates_mode1_output/groups.txt \
  || die "Unexpected differences in groups.txt output of hltFindDuplicates (mode: 1)" $?
