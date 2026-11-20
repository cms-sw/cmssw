#! /bin/bash

# Check that the MPIController has started processing on all its streams,
# across different runs and luminosity sections, before the MPISource starts
# processing its first event.

function usage() {
  echo "usage:"
  echo "  $(basename $0) controller_cfg.py follower_cfg.py [nStreams]"
  echo
}

if [ $# -lt 2 ]; then
  usage
  echo "$(basename $0): error: please specify the controller and follower configurations."
  exit 1
fi

CONTROLLER=$(realpath "$1")
FOLLOWER=$(realpath "$2")
EXPECTED_STREAMS=${3:-10}

if ! [ -f "$CONTROLLER" ]; then
  echo "$(basename $0): error: invalid argument '$CONTROLLER'"
  exit 1
fi

if ! [ -f "$FOLLOWER" ]; then
  echo "$(basename $0): error: invalid argument '$FOLLOWER'"
  exit 1
fi

# Make sure the CMSSW environment has been loaded.
if [ -z "$CMSSW_BASE" ]; then
  eval `scram runtime -sh`
fi

# The CM pml leads to silent communication failures on some machines.
# Until this is understood and fixed, keep it disabled.
export OMPI_MCA_pml='^cm'

mkdir -p $CMSSW_BASE/tmp/$SCRAM_ARCH/test
DIR=$(mktemp -d -p $CMSSW_BASE/tmp/$SCRAM_ARCH/test)
echo "Running MPI stream-ordering test at $DIR/"
pushd $DIR > /dev/null

# Run the controller and the follower in the same MPI_COMM_WORLD.
# --tag-output prefixes every line with [job,rank]<stdout>: / <stderr>:
{
  timeout --signal=SIGTERM 120 \
    mpirun --tag-output -n 1 cmsRun "$CONTROLLER" : -n 1 cmsRun "$FOLLOWER" >& mpi.log
  echo $? > mpi.status
}

MPI_STATUS=$(< mpi.status)

# Print the output and check the run status.
echo '========== testMPIConcurrentRunLumi ==========='
cat mpi.log
echo '============================================'
echo

if [ "$MPI_STATUS" -ne 0 ]; then
  echo "$(basename $0): error: mpirun/cmsRun failed with status $MPI_STATUS"
  popd > /dev/null
  exit $MPI_STATUS
fi

# Check that:
#   - the controller has started at least one event per stream before the
#     source starts;
#   - the first nStreams controller lines cover all controller streams;
#   - those lines contain more than one run and more than one lumi section.
awk -v expected_arg="$EXPECTED_STREAMS" '
function number_after(s, key,    p) {
  if (match(s, key "[0-9]+")) {
    p = substr(s, RSTART + length(key), RLENGTH - length(key))
    return p + 0
  }
  return -1
}

{
  if (!match($0, /^\[[0-9]+,[0-9]+\]/))
    next

  tag = substr($0, RSTART, RLENGTH)
  msg = substr($0, RLENGTH + 1)
  sub(/^<std(out|err)>: ?/, "", msg)

  # Extract [job,rank] and use the rank to identify the process.
  t = tag
  gsub(/^\[/, "", t)
  gsub(/\]$/, "", t)
  split(t, parts, ",")
  rank = parts[2] + 0

  if (rank == 0)
    is_controller[tag] = 1
  else
    is_source[tag] = 1

  # The controller logs its actual number of streams.
  if (rank == 0 && msg ~ /^setting # streams [0-9]+/)
    detected_streams = number_after(msg, "setting # streams ")

  if (msg ~ /^Begin processing the / &&
      msg ~ /Run [0-9]+/ &&
      msg ~ /LumiSection [0-9]+/ &&
      msg ~ /on stream [0-9]+/) {
    ++nbegin
    begin_tag[nbegin]    = tag
    begin_run[nbegin]    = number_after(msg, "Run ")
    begin_lumi[nbegin]   = number_after(msg, "LumiSection ")
    begin_stream[nbegin] = number_after(msg, "on stream ")
    begin_msg[nbegin]    = msg
  }
}

END {
  if (detected_streams > 0)
    expected = detected_streams
  else if (expected_arg > 0)
    expected = expected_arg
  else {
    print "ERROR: could not determine the controller stream count" > "/dev/stderr"
    exit 2
  }

  ncontroller = 0
  for (t in is_controller)
    ++ncontroller

  nsource = 0
  for (t in is_source)
    ++nsource

  if (ncontroller == 0) {
    print "ERROR: could not identify the controller output in the tagged log" > "/dev/stderr"
    exit 3
  }

  if (nsource == 0) {
    print "ERROR: could not identify the source output in the tagged log" > "/dev/stderr"
    exit 4
  }

  first_source = 0
  for (i = 1; i <= nbegin; ++i) {
    if (is_source[begin_tag[i]]) {
      first_source = i
      break
    }
  }

  if (first_source == 0) {
    print "ERROR: no Source \"Begin processing\" line found" > "/dev/stderr"
    exit 5
  }

  controller_before = 0
  for (i = 1; i < first_source; ++i) {
    if (is_controller[begin_tag[i]])
      ++controller_before
  }

  if (controller_before < expected) {
    printf "ERROR: only %d Controller \"Begin processing\" line(s) before the first Source line; expected at least %d (one per controller stream)\n", controller_before, expected > "/dev/stderr"
    printf "First Source line: %s\n", begin_msg[first_source] > "/dev/stderr"
    exit 6
  }

  seen = 0
  for (i = 1; i < first_source && seen < expected; ++i) {
    if (!is_controller[begin_tag[i]])
      continue

    ++seen
    streams[begin_stream[i]] = 1
    runs[begin_run[i]]       = 1
    lumis[begin_lumi[i]]     = 1
  }

  nstreams = 0
  for (s in streams)
    ++nstreams

  nruns = 0
  for (r in runs)
    ++nruns

  nlumis = 0
  for (l in lumis)
    ++nlumis

  if (nstreams != expected) {
    printf "ERROR: first %d Controller lines cover %d distinct stream(s); expected %d\n", expected, nstreams, expected > "/dev/stderr"
    exit 7
  }

  if (nruns < 2) {
    printf "ERROR: first %d Controller lines use only %d run; expected more than one run\n", expected, nruns > "/dev/stderr"
    exit 8
  }

  if (nlumis < 2) {
    printf "ERROR: first %d Controller lines use only %d lumi section; expected more than one lumi section\n", expected, nlumis > "/dev/stderr"
    exit 9
  }

  printf "OK: %d Controller Begin-processing lines occurred before the Source started\n", controller_before
  printf "OK: the first %d Controller lines covered %d distinct streams, %d runs and %d lumi sections\n", expected, nstreams, nruns, nlumis
  printf "First Source line: %s\n", begin_msg[first_source]
  exit 0
}
' mpi.log

CHECK_STATUS=$?

popd > /dev/null
exit $CHECK_STATUS
