#!/usr/bin/env bash

# Submit one independent OO2025 DATA task per dataset.
# Continue after individual failures and report a nonzero final status.
set -u

usage() {
    cat <<'USAGE'
Usage: submit_multidataset_OO_DATA.sh [--dry-run] [DATASET_FILE]

Default dataset list: datasets_OO_DATA.txt beside this script.
Custom relative dataset-list paths are resolved from the caller's directory.
  --dry-run   Print commands without invoking CRAB or creating tasks.
  -h, --help  Show this help.
USAGE
}

dry_run=false
dataset_file=""
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=true ;;
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

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
if [[ -z "${dataset_file}" ]]; then
    dataset_file="${script_directory}/datasets_OO_DATA.txt"
elif [[ "${dataset_file}" != /* ]]; then
    dataset_file="${PWD}/${dataset_file}"
fi

# Keep config, pset, mask and project paths consistent for any caller.
cd -- "${script_directory}" || exit 1

CRAB_CONFIG="crab_OO_DATA.py"
WORK_AREA="crab_projects/OO2025_DATA_Nebraska_slim"
REQUEST_PREFIX="HiForest_OO2025_DATA"
REQUEST_VERSION="Nebraska_slim_v1"

if [[ ! -f "${CRAB_CONFIG}" ]]; then
    echo "ERROR: CRAB configuration not found: ${CRAB_CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${dataset_file}" || ! -r "${dataset_file}" ]]; then
    echo "ERROR: Dataset list not found or not readable: ${dataset_file}" >&2
    exit 1
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
    if [[ ! "${dataset}" =~ ^/IonPhysics[0-9]+/OORun2025-PromptReco-v1/MINIAOD$ ]]; then
        echo "[FAILED] Invalid dataset entry: ${dataset}" >&2
        failed=$((failed + 1))
        continue
    fi

    primary_dataset="${dataset#/}"
    primary_dataset="${primary_dataset%%/*}"
    request_name="${REQUEST_PREFIX}_${primary_dataset}_${REQUEST_VERSION}"
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
        "Data.outLFNDirBase=/store/user/jvelazqu/HiForest/OO2025/DATA"
        "Site.storageSite=T2_US_Nebraska"
    )

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

echo "OO2025 DATA CRAB submission summary"
echo "Planned: ${planned}"
echo "Submitted: ${submitted}"
echo "Skipped: ${skipped}"
echo "Failed: ${failed}"
[[ "${failed}" -eq 0 ]]
