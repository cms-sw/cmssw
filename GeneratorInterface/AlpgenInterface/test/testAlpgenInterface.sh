#!/bin/bash

# This is going to be a script that performs a test suit over the 
# AlpgenInterface package. Still in development.
# Any questions, please contact Thiago Tomei (IFT-UNESP / SPRACE). 

# Modify these defaults to suit your environment.
ARCHITECTURE=$SCRAM_ARCH
ALPGEN_PATH=$CMS_PATH/sw/$ARCHITECTURE/external/alpgen/213-cms
ALPGEN_BIN_PATH=$ALPGEN_PATH/bin

# Hi!
echo "++++++++++++++++++++"
echo "+ ALPGEN INTERFACE +"
echo "++++++++++++++++++++"
echo "" 
echo "Starting basic sanity checks for Alpgen in CMSSW" 
echo ""
# See the available executables.
echo "The Alpgen executables should be found in $ALPGEN_BIN_PATH"
echo "Available executables in $ALPGEN_BIN_PATH:"
ls $ALPGEN_BIN_PATH
echo ""

# Create an empty input file.
if ls input.skel &> /dev/null
  then \rm input.skel
fi
touch input.skel

# Some functions that we define for later use

# A function to create the input file using a here document.
create_input_file ()
{
# First, setup derived parameters.
INPUTFILENAME=$LABEL.input
STDOUTFILENAME=$LABEL.stdout
if [[ "$ICKKW" -eq 1 ]]
then 
  MATCHINGPYTHON=True
else
  MATCHINGPYTHON=False
fi
ETCLUS=$(bc <<CalculationLimitString
scale = 4
var1=$ETMIN+5.0
var2=1.2*$ETMIN
if(var1 > var2) var1
if(var1 <= var2) var2
CalculationLimitString
)

# Now, write the file proper.
cat > input.skel <<EOF
$IMODE         ! imode
$LABEL       ! label for files
0         ! start with: 0=new grid, 1=previous warmup grid, 2=previous generation grid
10000 2   ! Nevents/iteration,  N(warm-up iterations)
100000    ! Nevents generated after warm-up
*** The above 5 lines provide mandatory inputs for all processes
*** (Comment lines are introduced by the three asteriscs)
*** The lines below modify existing defaults for the hard process under study
*** For a complete list of accessible parameters and their values,
*** input 'print 1' (to display on the screen) or 'print 2' to write to file
ih2 $IH2                ! nature of collisions: pp (1) or ppbar (-1)
ebeam $ENERGY           ! beam energy in GeV
ickkw $ICKKW              ! matching on (1) or off (0)
etajmax 5            ! full rap range for jets
njets   $NJETS            ! total number of jets
ptjmin  $ETMIN         ! ptmin for jets
drjmin  $DRMIN          ! minimum separation for jets
EOF
mv input.skel $INPUTFILENAME
}

# Function to run a test.
run_test()
{
# Create the ALPGEN input file (default: W+1j at the Tevatron).
IMODE=1
create_input_file
# Run it.
echo "Testing: $ALPGEN_BIN_PATH/$ALPGEN_EXECUTABLE < $INPUTFILENAME (imode 1)..."
$ALPGEN_BIN_PATH/$ALPGEN_EXECUTABLE < $INPUTFILENAME > $STDOUTFILENAME

# Also for imode = 2.
IMODE=2
create_input_file
echo "Testing: $ALPGEN_BIN_PATH/$ALPGEN_EXECUTABLE < $INPUTFILENAME (imode 2)..."
echo "++++++++++" >> $STDOUTFILENAME
$ALPGEN_BIN_PATH/$ALPGEN_EXECUTABLE < $INPUTFILENAME >> $STDOUTFILENAME

# Create the cmsRun cfg.py file.
cat test_TEMPLATE_cfg.py | sed "s/FILENAME/$LABEL/" | sed "s/MATCHING/$MATCHINGPYTHON/" \
| sed "s/EXCLUSIVE/$EXCLUSIVE/" | sed "s/ETMIN/$ETCLUS/" | sed "s/DRMIN/$DRMIN/" > test_cfg.py
# Run it.
echo "Testing: cmsRun test_cfg.py (running both AlpgenSource and AlpgenProducer)..."
cmsRun test_cfg.py > cmsRun.stdout 2> cmsRun.stderr
}

# Function to display the test results.
display_test_results()
{
# The input file.
echo -e "\nINPUT FILE: $INPUTFILENAME"
cat $INPUTFILENAME
# Beam content.
if [[ "$IH2" -eq 1 ]]
then BEAMCONTENT="proton proton" 
elif [[ "$IH2" -eq -1  ]]
then BEAMCONTENT="proton antiproton"
else BEAMCONTENT="unkwown"
fi
echo -e "\nBeam content given as input: $BEAMCONTENT"
echo "Beam content seem by CMSSW: "
grep "PYTHIA will be initialized" cmsRun.stdout
# Beam energy.
echo -e "\nBeam energy given as input: $ENERGY GeV"
echo -n "Beam energy seem by CMSSW: "
grep energies cmsRun.stdout | awk '{print $3,$4}'
# Jet parameters for matching.
echo -e "\nJet parameters for matching"
echo "Given as input: EXCLUSIVE = $EXCLUSIVE"
echo "                ETCLUS = $ETCLUS"
echo "                DRMIN = $DRMIN"
echo "Seen by CMSSW:"
grep IEXC cmsRun.stdout
grep ETACLUS cmsRun.stdout
echo "===================="
echo
}

# Parameters for the ALPGEN input file.
# Also, the exclusive / inclusive option.
# Must be either True or False
ALPGEN_EXECUTABLE=wjetgen
LABEL=w1j
IH2=-1
ENERGY=980.0
ICKKW=1
NJETS=1
ETMIN=20.0
DRMIN=0.7
EXCLUSIVE=True
# Run test and display test results
run_test
display_test_results

# Now you just change the values and run new tests.
# Test Z+jets
ALPGEN_EXECUTABLE=zjetgen
LABEL=z0j
IH2=1
ENERGY=7000.0
ICKKW=0
NJETS=0
ETMIN=20.0
DRMIN=0.7
EXCLUSIVE=False
run_test
display_test_results

# Test bbbar+jets
ALPGEN_EXECUTABLE=2Qgen
LABEL=bbbar1j
IH2=1
ENERGY=5000.0
ICKKW=1
NJETS=1
ETMIN=40.0
DRMIN=0.7
EXCLUSIVE=True
run_test
display_test_results
