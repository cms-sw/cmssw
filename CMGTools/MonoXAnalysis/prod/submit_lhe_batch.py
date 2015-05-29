#! /usr/bin/env python
# usage: ./submit_lhe_batch.py -T 100000 -N 10000 --eos=/store/cmst3/user/emanuele/monox/mcsamples/zjet_8tev/gen/gamma1jet_LO_pT100 /afs/cern.ch/work/a/avartak/public/ZGammaFiles/gamma1jet_LO_pT100.lhe

import os
import sys
import optparse

def main():
    parser = optparse.OptionParser()
    parser.add_option('-T', '--eventsperfile', action='store',     dest='eventsperfile',  help='number of events per input file'                     , default=2000,   type='int')
    parser.add_option('-N', '--neventsjob', action='store',     dest='neventsjob',  help='split the jobs with n events  / batch job'                 , default=200,   type='int')
    parser.add_option('--eos',               action='store',     dest='eos',         help='copy the output in the specified EOS path'                 , default='')

    (opt, args) = parser.parse_args()

    if len(args) != 1:
        print usage
        sys.exit(1)
    lhefile = args[0]
    basename = os.path.basename(lhefile)
    sample = os.path.splitext(basename)[0]
       
    ijob=0

    pwd = os.environ['PWD']

    firstEvent = 1
    firstLuminosityBlock = 1
    while (firstEvent < opt.eventsperfile or opt.eventsperfile == -1):
        lastEvent = firstEvent+opt.neventsjob
        os.system("echo bsub -q 8nh "+pwd+"/lhe2gen2.sh --events "+str(firstEvent-1)+" "+str(+opt.neventsjob)+" --firstLuminosityBlock "+str(ijob+1)+" gen_z1jet.py "+opt.eos+"/"+sample+"_"+str(ijob)+".root "+lhefile)
        os.system("bsub -q 8nh "+pwd+"/lhe2gen2.sh --events "+str(firstEvent-1)+" "+str(+opt.neventsjob)+" --firstLuminosityBlock "+str(ijob+1)+" gen_z1jet.py "+opt.eos+"/"+sample+"_"+str(ijob)+".root "+lhefile)
        ijob = ijob+1
        if (opt.eventsperfile == -1): break
        else: firstEvent = lastEvent

if __name__ == "__main__":
        main()

