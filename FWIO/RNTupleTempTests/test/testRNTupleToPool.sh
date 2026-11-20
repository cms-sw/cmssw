#!/bin/bash
set -e
CFG="${SCRAM_TEST_PATH}/testRNTupleToPool_cfg.py"

function run {
  local NAME="$1"
  shift
  cmsRun -j "${NAME}.xml" "$CFG" "$@" || {
    echo "${NAME} failed"
    return 1
  }
}

run write_rntuple --step rntuple --output rntuple_input.root
run convert_rntuple_to_pool --step convert --input rntuple_input.root --output rntuple_to_pool.root
run read_rntuple_to_pool --step read --input rntuple_to_pool.root
run drop_rntuple_to_pool --step drop --input rntuple_to_pool.root
run write_pool --step emptyPool --output empty_to_pool.root
run read_pool --step read --input empty_to_pool.root
run drop_pool --step drop --input empty_to_pool.root
run copy_pool --step copy --input empty_to_pool.root --output pool_copy.root
run read_pool_copy --step read --input pool_copy.root

python3 - <<'PY'
import xml.etree.ElementTree as ET

for filename, expected, input_pfn, input_class in (
    ("convert_rntuple_to_pool.xml", "0", "file:rntuple_input.root", "RNTupleTempSource"),
    ("copy_pool.xml", "1", "file:empty_to_pool.root", "PoolSource"),
):
    root = ET.parse(filename).getroot()
    outputs = root.findall("./File")
    inputs = root.findall("./InputFile")
    if len(outputs) != 1 or len(inputs) != 1:
        raise RuntimeError(f"{filename}: expected one input and one output")
    if inputs[0].findtext("PFN") != input_pfn or inputs[0].findtext("InputSourceClass") != input_class:
        raise RuntimeError(f"{filename}: unexpected input file or source")
    if outputs[0].findtext("TotalEvents") != "3" or outputs[0].findtext("./Inputs/Input/PFN") != input_pfn:
        raise RuntimeError(f"{filename}: unexpected output event count or input association")
    statuses = [entry.text for output in outputs for entry in output.findall("./Inputs/Input/FastCopying")]
    if statuses != [expected]:
        raise RuntimeError(f"{filename}: expected fast-copy status {[expected]}, got {statuses}")

for filename in ("convert_rntuple_to_pool.xml", "read_rntuple_to_pool.xml", "drop_rntuple_to_pool.xml",
                 "read_pool.xml", "drop_pool.xml", "copy_pool.xml", "read_pool_copy.xml"):
    root = ET.parse(filename).getroot()
    counts = [entry.text for entry in root.findall("./InputFile/EventsRead")]
    if counts != ["3"]:
        raise RuntimeError(f"{filename}: expected three input events, got {counts}")
PY
