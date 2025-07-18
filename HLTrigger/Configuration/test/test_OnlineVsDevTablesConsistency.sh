#!/bin/bash

function die { echo $1: status $2 ; exit $2; }

check_file_existence() {
    local file_path="$1"

    if [ -e "$file_path" ]; then
        # File exists, do nothing
        :
    else
        # Print error message in red
        echo -e "\e[91mError: File '$file_path' does not exist.\e[0m"
        exit 1
    fi
}

function compare_files() {
    local file1_path="$1"
    local file2_path="$2"

    ## check that the input files exist
    check_file_existence "$file1_path"
    check_file_existence "$file2_path"

    local exclude_set=("HLTAnalyzerEndpath" "RatesMonitoring" "DQMHistograms")

    local lines_file1=()
    local lines_file2=()
    local not_in_file2=()

    # extract the list of paths to match from the first file
    while IFS= read -r line; do
    if [[ ! $line =~ ^# ]]; then
        # Extract the first word before a space
        first_word=$(echo "$line" | awk '{print $1}')
        lines_file1+=("$first_word")
    fi
    done < "$file1_path"

    # extract the list of paths to match from the second file
    while IFS= read -r line; do
    if [[ ! $line =~ ^# ]]; then
        # Extract the first word before a space
        first_word=$(echo "$line" | awk '{print $1}')
        lines_file2+=("$first_word")
    fi
    done < "$file2_path"

    # find the set not in common
    for line in "${lines_file1[@]}"; do
        if [[ ! "${lines_file2[@]}" =~ "$line" ]]; then
            not_in_file2+=("$line")
        fi
    done

    # Remove lines from not_in_file2 that contain any substring in exclude_set
    for pattern in "${exclude_set[@]}"; do
        not_in_file2=("${not_in_file2[@]//*$pattern*}")
    done

    # Remove empty elements and empty lines after substitution
    not_in_file2=("${not_in_file2[@]//''}")

    # Remove empty elements from the array
    local cleaned_not_in_file2=()
    for element in "${not_in_file2[@]}"; do
        if [[ -n "$element" ]]; then
            cleaned_not_in_file2+=("$element")
        fi
    done

    file1_name=$(basename "$file1_path")
    file2_name=$(basename "$file2_path")
    
    if [ ${#cleaned_not_in_file2[@]} -eq 0 ]; then
        echo -e "\033[92mAll lines from $file1_name are included in $file2_name.\033[0m"
        return 0
    else
        echo "Lines present in $file1_name but not in $file2_name (excluding the exclusion set):"
        printf '%s\n' "${not_in_file2[@]}"
        return 1
    fi
}

TABLES_AREA="$CMSSW_BASE/src/HLTrigger/Configuration/tables"

compare_files $TABLES_AREA/online_pion.txt $TABLES_AREA/PIon.txt || die "Failure comparing online_pion and PIon" $?
compare_files $TABLES_AREA/online_hion.txt $TABLES_AREA/HIon.txt  || die "Failure comparing online_hion and HIon" $?
compare_files $TABLES_AREA/online_pref.txt $TABLES_AREA/PRef.txt  || die "Failure comparing online_pref and PRef" $?
compare_files $TABLES_AREA/online_Circulating.txt  $TABLES_AREA/Special.txt  || die "Failure comparing online_Circulating and Special" $?
compare_files $TABLES_AREA/online_PPS.txt $TABLES_AREA/Special.txt  || die "Failure comparing online_PPS and Special" $?
compare_files $TABLES_AREA/online_LumiScan.txt $TABLES_AREA/Special.txt  || die "Failure comparing online_LumiScan and Special" $?
compare_files $TABLES_AREA/online_FirstCollisions.txt $TABLES_AREA/Special.txt  || die "Failure comparing online_FirstCollisions and Special" $?
compare_files $TABLES_AREA/online_ECAL.txt $TABLES_AREA/Special.txt  || die "Failure comparing online_ECAL and Special" $?
compare_files $TABLES_AREA/online_Cosmics.txt $TABLES_AREA/Special.txt  || die "Failure comparing online_Cosmics and Special" $?
compare_files $TABLES_AREA/online_TrackerVR.txt $TABLES_AREA/Special.txt  || die "Failure comparing online_TrackerVR and Special" $?
compare_files $TABLES_AREA/online_Splashes.txt $TABLES_AREA/Special.txt  || die "Failure comparing online_Splashes and Special" $?
compare_files $TABLES_AREA/online_Special.txt $TABLES_AREA/Special.txt  || die "Failure comparing online_Special and Special" $?
compare_files $TABLES_AREA/online_grun.txt $TABLES_AREA/GRun.txt  || die "Failure comparing online_grun and GRun" $?
