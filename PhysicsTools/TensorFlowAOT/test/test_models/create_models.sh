#!/usr/bin/env bash

# Script to create simple AOT compiled models for testing purposes.

action() {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"
    local aot_dir="${CMSSW_BASE}/src/PhysicsTools/TensorFlowAOT"

    # remove existing models
    rm -rf "${this_dir}"/*model

    # create saved models
    python3 -W ignore "${aot_dir}/test/create_model.py" \
        --model-dir "${this_dir}/saved_simplemodel"\
        || return "$?"
    python3 -W ignore "${aot_dir}/test/create_model.py" \
        --model-dir "${this_dir}/saved_multimodel" \
        --multi-tensor \
        || return "$?"

    # comple them
    python3 -W ignore "${aot_dir}/scripts/compile_model.py" \
        --model-path "${this_dir}/saved_simplemodel" \
        --model-name "simplemodel" \
        --subsystem PhysicsTools \
        --package TensorFlowAOT \
        --batch-sizes "1,2" \
        --output-path "${this_dir}/simplemodel" \
        || return "$?"
    python3 -W ignore "${aot_dir}/scripts/compile_model.py" \
        --model-path "${this_dir}/saved_multimodel" \
        --model-name "multimodel" \
        --subsystem PhysicsTools \
        --package TensorFlowAOT \
        --batch-sizes "1,2" \
        --output-path "${this_dir}/multimodel" \
        || return "$?"

    # remove custom tool files
    rm "${this_dir}/"*model/tfaot-dev-physicstools-tensorflowaot-*model.xml || return "$?"

    # remove saved models again
    rm -rf "${this_dir}"/saved_*model
}
action "$@"
