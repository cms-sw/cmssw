#!/usr/bin/env bash

# Submit one independent ppref2024 DATA task per dataset.
# Continue after individual failures and report a nonzero final status.
set -u

usage() {
    cat <<'USAGE'
Usage: submit_multidataset_ppref_DATA.sh [--dry-run] [--zero-bias] [--discover | DATASET_FILE]

Default dataset list: datasets_ppref_DATA.txt beside this script.
Custom relative dataset-list paths are resolved from the caller's directory.
  --dry-run   Print commands without invoking CRAB or creating tasks.
  --discover  Query DAS for all PPRef Run2024J PromptReco-v1 MINIAOD datasets.
  --zero-bias Submit only PPRefZeroBiasPlusForward0 and 1 from the dataset list.
  -h, --help  Show this help.
USAGE
}

dry_run=false
discover=false
zero_bias=false
dataset_file=""
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=true ;;
        --discover) discover=true ;;
        --zero-bias) zero_bias=true ;;
        -h|--help) usage; exit 0 ;;
        -*)
            echo "ERROR: Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -n "${dataset_file}" || -z "$1" ]]; then
                echo "ERROR: Provide exactly one nonempty dataset-list path." >&2
                exit 1
            fi
            dataset_file="$1"
            ;;
    esac
    shift
done

if "${discover}" && [[ -n "${dataset_file}" ]]; then
    echo "ERROR: --discover and DATASET_FILE cannot be combined." >&2
    exit 1
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
if "${discover}"; then
    if ! command -v dasgoclient >/dev/null 2>&1; then
        echo "ERROR: dasgoclient is required for --discover." >&2
        exit 1
    fi
    dataset_file="$(mktemp)" || exit 1
    trap 'rm -f -- "$dataset_file"' EXIT
    if ! dasgoclient -query='dataset=/PPRef*/Run2024J-PromptReco-v1/MINIAOD' -limit=0 > "${dataset_file}"; then
        echo "ERROR: DAS query failed; no CRAB tasks submitted." >&2
        exit 1
    fi
    if [[ ! -s "${dataset_file}" ]]; then
        echo "ERROR: DAS query returned an empty list; no CRAB tasks submitted." >&2
        exit 1
    fi
    if ! grep -Fxq '/PPRefZeroBiasPlusForward1/Run2024J-PromptReco-v1/MINIAOD' "${dataset_file}"; then
        echo "ERROR: DAS did not return PPRefZeroBiasPlusForward1; no CRAB tasks submitted." >&2
        exit 1
    fi
elif [[ -z "${dataset_file}" ]]; then
    dataset_file="${script_directory}/datasets_ppref_DATA.txt"
elif [[ "${dataset_file}" != /* ]]; then
    dataset_file="${PWD}/${dataset_file}"
fi

# Keep config, pset, mask and project paths consistent for any caller.
cd -- "${script_directory}" || exit 1

CRAB_CONFIG="crab_ppref_DATA.py"
WORK_AREA="crab_projects/ppref2024_DATA_Nebraska"
REQUEST_PREFIX="HiForest_ppref2024_DATA"
REQUEST_VERSION="Nebraska_v1"

if [[ ! -f "${CRAB_CONFIG}" ]]; then
    echo "ERROR: CRAB configuration not found: ${CRAB_CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${dataset_file}" || ! -r "${dataset_file}" ]]; then
    echo "ERROR: Dataset list not found or not readable: ${dataset_file}" >&2
    exit 1
fi
if "${zero_bias}"; then
    for stream in 0 1; do
        expected="/PPRefZeroBiasPlusForward${stream}/Run2024J-PromptReco-v1/MINIAOD"
        if ! grep -Fxq "${expected}" "${dataset_file}"; then
            echo "ERROR: Dataset list is missing ${expected}; no CRAB tasks submitted." >&2
            exit 1
        fi
    done
fi
if ! "${dry_run}" && ! command -v crab >/dev/null 2>&1; then
    echo "ERROR: crab is not available; initialize the CRAB environment." >&2
    exit 1
fi

submitted=0
planned=0
skipped=0
failed=0
declare -A seen_requests=()

while IFS= read -r dataset || [[ -n "${dataset}" ]]; do
    dataset="${dataset%$'\r'}"
    [[ "${dataset}" =~ ^[[:space:]]*$ ]] && continue
    [[ "${dataset}" =~ ^[[:space:]]*# ]] && continue

    # Accept dataset names, not /store/... ROOT filenames.
    if [[ ! "${dataset}" =~ ^/PPRef[A-Za-z0-9_]+/Run2024J-PromptReco-v1/MINIAOD$ ]]; then
        echo "[FAILED] Invalid dataset entry: ${dataset}" >&2
        failed=$((failed + 1))
        continue
    fi

    primary_dataset="${dataset#/}"
    primary_dataset="${primary_dataset%%/*}"
    if "${zero_bias}" && [[ ! "${primary_dataset}" =~ ^PPRefZeroBiasPlusForward[01]$ ]]; then
        continue
    fi

    # These streams each generated ~17,800 jobs at 50,000 events/job.
    # Use a fresh request name and larger splitting target for the retries.
    request_version="${REQUEST_VERSION}"
    units_per_job=""
    if [[ "${primary_dataset}" =~ ^PPRefZeroBiasPlusForward[01]$ ]]; then
        request_version="Nebraska_v2"
        units_per_job="125000"
    fi

    # Do not change the dataset name after extracting its primary stream.
    request_name="${REQUEST_PREFIX}_${primary_dataset}_${request_version}"
    project_directory="${WORK_AREA}/crab_${request_name}"

    if [[ -n "${seen_requests[${request_name}]:-}" ]]; then
        echo "[FAILED] Duplicate request name in dataset list: ${request_name}" >&2
        failed=$((failed + 1))
        continue
    fi
    seen_requests["${request_name}"]=1

    if [[ -d "${project_directory}" ]]; then
        echo "[SKIPPED] Local project already exists: ${project_directory}"
        skipped=$((skipped + 1))
        continue
    fi

    command=(
        crab submit
        -c "${CRAB_CONFIG}"
        "General.workArea=${WORK_AREA}"
        "General.requestName=${request_name}"
        "Data.inputDataset=${dataset}"
        "Data.outputDatasetTag=${request_name}"
        "Data.outLFNDirBase=/store/user/jvelazqu/HiForest/ppref2024/DATA"
        "Site.storageSite=T2_US_Nebraska"
    )

    if [[ -n "${units_per_job}" ]]; then
        command+=("Data.unitsPerJob=${units_per_job}")
    fi

    if "${dry_run}"; then
        printf '[DRY RUN] '
        printf '%q ' "${command[@]}"
        printf '\n'
        planned=$((planned + 1))
    elif "${command[@]}"; then
        echo "[SUCCESS] Submitted ${dataset}"
        submitted=$((submitted + 1))
    else
        echo "[FAILED] Submission failed for ${dataset}" >&2
        failed=$((failed + 1))
    fi
done < "${dataset_file}"

echo "ppref2024 DATA CRAB submission summary"
echo "Planned: ${planned}"
echo "Submitted: ${submitted}"
echo "Skipped: ${skipped}"
echo "Failed: ${failed}"
[[ "${failed}" -eq 0 ]]
