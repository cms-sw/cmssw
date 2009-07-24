#!/bin/sh

echo ""
echo "========================================"
echo "Setting run numbers..."
echo "========================================"
./setRunNumbers.sh
echo ""

./sendCmdToAllApps.sh Enable
