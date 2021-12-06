#!/bin/bash
check_for_success() {
    "${@}" && echo -e "\n ---> Passed test of '${@}'\n\n" || exit 1
}

check_for_failure() {
    "${@}" && exit 1 || echo -e "\n ---> Passed test of '${@}'\n\n"
}

function die { echo $1: status $2; exit $2; }

########################################
# Test help function
########################################
check_for_success uploadConditions.py --help

########################################
# Test wizard
########################################
if test -f "BasicPayload_v0_ref.txt"; then
    rm -f BasicPayload_v0_ref.txt
fi
cat <<EOF >> BasicPayload_v0_ref.txt
{
    "destinationDatabase": "oracle://cms_orcoff_prep/CMS_CONDITIONS", 
    "destinationTags": {
        "BasicPayload_v0": {}
    }, 
    "inputTag": "BasicPayload_v0", 
    "since": 1, 
    "userText": "uploadConditions unit test"
}
EOF

echo "Content of the directory is:" `ls -lh . | grep db`
echo -ne '\n\n'

if test -f "BasicPayload_v0.txt"; then
   rm -f BasicPayload_v0.txt
fi

# this is expected to fail given lack of credentials
check_for_failure uploadConditions.py BasicPayload_v0.db <<EOF
y
0
oracle://cms_orcoff_prep/CMS_CONDITIONS
1
uploadConditions unit test
BasicPayload_v0
`echo -ne '\n'`
y
test
test
EOF

# test that the metadata created with the wizard corresponds to the reference one
diff -w BasicPayload_v0.txt BasicPayload_v0_ref.txt || die 'failed comparing metadata with reference' $?
