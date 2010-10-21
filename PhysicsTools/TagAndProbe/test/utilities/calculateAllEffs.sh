#!/bin/sh

echo "NOW CALCULATING ALL DATA AND MC EFFS"
#set up the root version -- DONT SET UP ANY OTHER ROOT VERSION BEFORE SETTING THIS UP!!!
source /uscmst1b_scratch/lpc1/old_scratch/lpctrig/jwerner/root/root-sl5/bin/thisroot.sh

root  -q -b -l effCalculator.C'(0,0,0,true,false)' > gsf.out
root  -q -b -l effCalculator.C'(0,1,0,true,false)' > gsf_eb.out
root  -q -b -l effCalculator.C'(0,2,0,true,false)' > gsf_ee.out
root  -q -b -l effCalculator.C'(0,1,1,true,false)' > gsf_eb_minus.out
root  -q -b -l effCalculator.C'(0,1,2,true,false)' > gsf_eb_plus.out
root  -q -b -l effCalculator.C'(0,2,1,true,false)' > gsf_ee_minus.out
root  -q -b -l effCalculator.C'(0,2,2,true,false)' > gsf_ee_plus.out


root  -q -b -l effCalculator.C'(1,0,0,true,false)' > id95.out
root  -q -b -l effCalculator.C'(1,1,0,true,false)' > id95_eb.out
root  -q -b -l effCalculator.C'(1,2,0,true,false)' > id95_ee.out
root  -q -b -l effCalculator.C'(1,1,1,true,false)' > id95_eb_minus.out
root  -q -b -l effCalculator.C'(1,1,2,true,false)' > id95_eb_plus.out
root  -q -b -l effCalculator.C'(1,2,1,true,false)' > id95_ee_minus.out
root  -q -b -l effCalculator.C'(1,2,2,true,false)' > id95_ee_plus.out


root  -q -b -l effCalculator.C'(2,0,0,true,false)' > id80.out
root  -q -b -l effCalculator.C'(2,1,0,true,false)' > id80_eb.out
root  -q -b -l effCalculator.C'(2,2,0,true,false)' > id80_ee.out
root  -q -b -l effCalculator.C'(2,1,1,true,false)' > id80_eb_minus.out
root  -q -b -l effCalculator.C'(2,1,2,true,false)' > id80_eb_plus.out
root  -q -b -l effCalculator.C'(2,2,1,true,false)' > id80_ee_minus.out
root  -q -b -l effCalculator.C'(2,2,2,true,false)' > id80_ee_plus.out

root  -q -b -l effCalculator.C'(3,0,0,true,false)' > hlt95.out
root  -q -b -l effCalculator.C'(3,1,0,true,false)' > hlt95_eb.out
root  -q -b -l effCalculator.C'(3,2,0,true,false)' > hlt95_ee.out
root  -q -b -l effCalculator.C'(3,1,1,true,false)' > hlt95_eb_minus.out
root  -q -b -l effCalculator.C'(3,1,2,true,false)' > hlt95_eb_plus.out
root  -q -b -l effCalculator.C'(3,2,1,true,false)' > hlt95_ee_minus.out
root  -q -b -l effCalculator.C'(3,2,2,true,false)' > hlt95_ee_plus.out

root  -q -b -l effCalculator.C'(4,0,0,true,false)' > hlt80.out
root  -q -b -l effCalculator.C'(4,1,0,true,false)' > hlt80_eb.out
root  -q -b -l effCalculator.C'(4,2,0,true,false)' > hlt80_ee.out
root  -q -b -l effCalculator.C'(4,1,1,true,false)' > hlt80_eb_minus.out
root  -q -b -l effCalculator.C'(4,1,2,true,false)' > hlt80_eb_plus.out
root  -q -b -l effCalculator.C'(4,2,1,true,false)' > hlt80_ee_minus.out
root  -q -b -l effCalculator.C'(4,2,2,true,false)' > hlt80_ee_plus.out


root  -q -b -l effCalculator.C'(0,0,0,false,false)' > gsf.out.mc
root  -q -b -l effCalculator.C'(0,1,0,false,false)' > gsf_eb.out.mc
root  -q -b -l effCalculator.C'(0,2,0,false,false)' > gsf_ee.out.mc
root  -q -b -l effCalculator.C'(0,1,1,false,false)' > gsf_eb_minus.out.mc
root  -q -b -l effCalculator.C'(0,1,2,false,false)' > gsf_eb_plus.out.mc
root  -q -b -l effCalculator.C'(0,2,1,false,false)' > gsf_ee_minus.out.mc
root  -q -b -l effCalculator.C'(0,2,2,false,false)' > gsf_ee_plus.out.mc


root  -q -b -l effCalculator.C'(1,0,0,false,false)' > id95.out.mc
root  -q -b -l effCalculator.C'(1,1,0,false,false)' > id95_eb.out.mc
root  -q -b -l effCalculator.C'(1,2,0,false,false)' > id95_ee.out.mc
root  -q -b -l effCalculator.C'(1,1,1,false,false)' > id95_eb_minus.out.mc
root  -q -b -l effCalculator.C'(1,1,2,false,false)' > id95_eb_plus.out.mc
root  -q -b -l effCalculator.C'(1,2,1,false,false)' > id95_ee_minus.out.mc
root  -q -b -l effCalculator.C'(1,2,2,false,false)' > id95_ee_plus.out.mc


root  -q -b -l effCalculator.C'(2,0,0,false,false)' > id80.out.mc
root  -q -b -l effCalculator.C'(2,1,0,false,false)' > id80_eb.out.mc
root  -q -b -l effCalculator.C'(2,2,0,false,false)' > id80_ee.out.mc
root  -q -b -l effCalculator.C'(2,1,1,false,false)' > id80_eb_minus.out.mc
root  -q -b -l effCalculator.C'(2,1,2,false,false)' > id80_eb_plus.out.mc
root  -q -b -l effCalculator.C'(2,2,1,false,false)' > id80_ee_minus.out.mc
root  -q -b -l effCalculator.C'(2,2,2,false,false)' > id80_ee_plus.out.mc


root  -q -b -l effCalculator.C'(3,0,0,false,false)' > hlt95.out.mc
root  -q -b -l effCalculator.C'(3,1,0,false,false)' > hlt95_eb.out.mc
root  -q -b -l effCalculator.C'(3,2,0,false,false)' > hlt95_ee.out.mc
root  -q -b -l effCalculator.C'(3,1,1,false,false)' > hlt95_eb_minus.out.mc
root  -q -b -l effCalculator.C'(3,1,2,false,false)' > hlt95_eb_plus.out.mc
root  -q -b -l effCalculator.C'(3,2,1,false,false)' > hlt95_ee_minus.out.mc
root  -q -b -l effCalculator.C'(3,2,2,false,false)' > hlt95_ee_plus.out.mc


root  -q -b -l effCalculator.C'(4,0,0,false,false)' > hlt80.out.mc
root  -q -b -l effCalculator.C'(4,1,0,false,false)' > hlt80_eb.out.mc
root  -q -b -l effCalculator.C'(4,2,0,false,false)' > hlt80_ee.out.mc
root  -q -b -l effCalculator.C'(4,1,1,false,false)' > hlt80_eb_minus.out.mc
root  -q -b -l effCalculator.C'(4,1,2,false,false)' > hlt80_eb_plus.out.mc
root  -q -b -l effCalculator.C'(4,2,1,false,false)' > hlt80_ee_minus.out.mc
root  -q -b -l effCalculator.C'(4,2,2,false,false)' > hlt80_ee_plus.out.mc


root  -q -b -l effCalculator.C'(0,0,0,false,true)' > gsf.out.mc.truth
root  -q -b -l effCalculator.C'(0,1,0,false,true)' > gsf_eb.out.mc.truth
root  -q -b -l effCalculator.C'(0,2,0,false,true)' > gsf_ee.out.mc.truth
root  -q -b -l effCalculator.C'(0,1,1,false,true)' > gsf_eb_minus.out.mc.truth
root  -q -b -l effCalculator.C'(0,1,2,false,true)' > gsf_eb_plus.out.mc.truth
root  -q -b -l effCalculator.C'(0,2,1,false,true)' > gsf_ee_minus.out.mc.truth
root  -q -b -l effCalculator.C'(0,2,2,false,true)' > gsf_ee_plus.out.mc.truth


root  -q -b -l effCalculator.C'(1,0,0,false,true)' > id95.out.mc.truth
root  -q -b -l effCalculator.C'(1,1,0,false,true)' > id95_eb.out.mc.truth
root  -q -b -l effCalculator.C'(1,2,0,false,true)' > id95_ee.out.mc.truth
root  -q -b -l effCalculator.C'(1,1,1,false,true)' > id95_eb_minus.out.mc.truth
root  -q -b -l effCalculator.C'(1,1,2,false,true)' > id95_eb_plus.out.mc.truth
root  -q -b -l effCalculator.C'(1,2,1,false,true)' > id95_ee_minus.out.mc.truth
root  -q -b -l effCalculator.C'(1,2,2,false,true)' > id95_ee_plus.out.mc.truth


root  -q -b -l effCalculator.C'(2,0,0,false,true)' > id80.out.mc.truth
root  -q -b -l effCalculator.C'(2,1,0,false,true)' > id80_eb.out.mc.truth
root  -q -b -l effCalculator.C'(2,2,0,false,true)' > id80_ee.out.mc.truth
root  -q -b -l effCalculator.C'(2,1,1,false,true)' > id80_eb_minus.out.mc.truth
root  -q -b -l effCalculator.C'(2,1,2,false,true)' > id80_eb_plus.out.mc.truth
root  -q -b -l effCalculator.C'(2,2,1,false,true)' > id80_ee_minus.out.mc.truth
root  -q -b -l effCalculator.C'(2,2,2,false,true)' > id80_ee_plus.out.mc.truth


root  -q -b -l effCalculator.C'(3,0,0,false,true)' > hlt95.out.mc.truth
root  -q -b -l effCalculator.C'(3,1,0,false,true)' > hlt95_eb.out.mc.truth
root  -q -b -l effCalculator.C'(3,2,0,false,true)' > hlt95_ee.out.mc.truth
root  -q -b -l effCalculator.C'(3,1,1,false,true)' > hlt95_eb_minus.out.mc.truth
root  -q -b -l effCalculator.C'(3,1,2,false,true)' > hlt95_eb_plus.out.mc.truth
root  -q -b -l effCalculator.C'(3,2,1,false,true)' > hlt95_ee_minus.out.mc.truth
root  -q -b -l effCalculator.C'(3,2,2,false,true)' > hlt95_ee_plus.out.mc.truth


root  -q -b -l effCalculator.C'(4,0,0,false,true)' > hlt80.out.mc.truth
root  -q -b -l effCalculator.C'(4,1,0,false,true)' > hlt80_eb.out.mc.truth
root  -q -b -l effCalculator.C'(4,2,0,false,true)' > hlt80_ee.out.mc.truth
root  -q -b -l effCalculator.C'(4,1,1,false,true)' > hlt80_eb_minus.out.mc.truth
root  -q -b -l effCalculator.C'(4,1,2,false,true)' > hlt80_eb_plus.out.mc.truth
root  -q -b -l effCalculator.C'(4,2,1,false,true)' > hlt80_ee_minus.out.mc.truth
root  -q -b -l effCalculator.C'(4,2,2,false,true)' > hlt80_ee_plus.out.mc.truth

root -q -b -l selectedEvents.C'(true, false)' > mc_effCrossCheck_ID95AND80.txt
root -q -b -l selectedEvents.C'(true, true)' > mc_effCrossCheck_SCtoGSF.txt

./parseOutput.sh

rm tpHistos_*_MONTECARLO.root
hadd -f tpHistos_ALL.root tpHistos_*.root 

echo "DONE CALCULATING EFFS -- OUTPUT DUMPED TO tpHistos_ALL.root AND efficiencyTable.txt"