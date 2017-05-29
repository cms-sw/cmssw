#! /bin/env bash

sed -i 's|Jet_eta|jetEta|g' $1
sed -i 's|Jet_pt|jetPt|g' $1
sed -i 's|TagVarCSV_||g' $1
sed -i 's|prob_|prob|g' $1

#bugfixes
sed -i 's|jetNTracks|jetNSelectedTracks|g' $1
sed -i 's|jetNSelectedTracksEtaRel|jetNTracksEtaRel|g' $1

python <<EOF
import json
with open('$1') as infile:
  jmap = json.loads(infile.read())

for var in jmap['inputs']:
  var['offset'] *= -1
  var['scale'] = 1./var['scale']

with open('$1', 'w') as out:
  out.write(json.dumps(jmap, indent=2, separators = (',', ': ')))
EOF