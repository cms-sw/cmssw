#! /bin/env bash

#sed -i 's|jet_eta|jetEta|g' $1
sed -i 's|jet_eta|jetAbsEta|g' $1
sed -i 's|jet_pt|jetPt|g' $1
sed -i 's|TagVarCSV_||g' $1
sed -i 's|TagVarCSVTrk_||g' $1
sed -i 's|prob_|prob|g' $1
sed -i 's|trackJetDistVal|trackJetDist|g' $1
