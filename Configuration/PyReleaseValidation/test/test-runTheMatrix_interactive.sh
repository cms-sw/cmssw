#!/bin/bash -e

echo 'help' | runTheMatrix.py --interactive || exit 1
echo 'predefined' | runTheMatrix.py --interactive || exit 1
echo 'showWorkflow' | runTheMatrix.py --interactive || exit 1
echo 'search .*D88.*' | runTheMatrix.py --interactive || exit 1
echo 'dumpWorkflowId 1.0' | runTheMatrix.py --interactive || exit 1
echo 'searchInWorkflow standard .*' | runTheMatrix.py --interactive || exit 1
echo 'wrongCommand' | runTheMatrix.py --interactive || exit 0

