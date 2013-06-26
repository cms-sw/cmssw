#! /bin/bash

# Navigate to the sample root files
cd /afs/cern.ch/cms/Tutorials/TWIKI_DATA/root/Rootplot

echo "Contents of hists_data.root:"
rootinfo hists_data.root
echo
echo "Contents of hists_qcd.root:"
rootinfo hists_qcd.root

echo
echo "Testing out rootplot with a few different options:"
rootplot hists_data.root --output=simple
rootplot hists_data.root -n --title="CMS Preliminary" --output=fancy
rootplot hists_data.root hists_qcd.root --output=compared

#### matplotlib version is currently incompatible with rootplot
## echo
## echo "Testing out rootplotmpl for output in matplotlib:"
## rootplotmpl hists_data.root hists_qcd.root
