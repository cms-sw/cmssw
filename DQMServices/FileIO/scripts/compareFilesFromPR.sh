#!/bin/bash
RESULTURL="$1"
BASELINEURL="$2"
PRNUMBER=$(date +%s)

fetch() {
  (
    export PATH=/bin/:/usr/bin/
    export PERL5LIB=
    export LD_LIBRARY_PATH=

    cern-get-sso-cookie -o cook --url $1
    for d in $(curl -L -s -k -b cook $1 | grep -oE '"[0-9]+*.[0-9]+_[^"]*"' | tr -d '"'); do
      for f in $(curl -L -s -k -b cook "$1/$d" | grep -oE '"DQM.*.root"' | tr -d '"'); do
        (echo "Fetching $d/$f..."; mkdir -p $d; cd $d; curl -O -L -s -k -b ../cook $1/$d/$f )
      done
    done
  )
}

if [[ -z $RESULTURL || -z $BASELINEURL ]]; then
  echo "Please provide a URL to 'Matrix Test Outputs', like 'https://cmssdt.cern.ch/SDT/jenkins-artifacts/pull-request-integration/PR-a20bd6/3666/runTheMatrix-results/', and a similar URL to use as the baseline, like 'https://cmssdt.cern.ch/SDT/jenkins-artifacts/ib-baseline-tests/CMSSW_11_0_X_2019-11-26-2300/slc7_amd64_gcc820/-GenuineIntel/matrix-results/'"
  echo "Requires cern-get-sso-cookie. This might only work *outside* a cmsenv."
  echo "Requires compareDQMOutput.py. This will only work *inside* a cmsenv."
  echo "You might need to run the script twice (before/after cmsenv) to get results."
  exit 1
fi

echo "Downloading PR files..."
mkdir -p pr
cd pr
fetch "$RESULTURL"
cd ..
echo "Downloading baseline files..."
mkdir -p base
cd base
fetch "$BASELINEURL"
cd ..

compareDQMOutput.py -b base/ -p pr/ -r "$CMSSW_VERSION"  -l "private/private#$PRNUMBER" -j12
