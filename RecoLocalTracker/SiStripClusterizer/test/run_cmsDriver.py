import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", dest="barycenter_bit", default='8bit', help="bit to be studied for barycenter")
parser.add_argument("-w", dest="width_bit", default='8bit', help="bit to be studied for width")
parser.add_argument("-a", dest="avgCharge_bit", default='8bit', help="bit to be studied for avgcharge")
parser.add_argument("-n", dest="number", default='500', help="how many numbers of events")
parser.add_argument("-t", dest="threads", default='20', help="how many threads")
parser.add_argument("-c", type=int, dest="cluster", default=1, help="want flatntuple for cluster")
parser.add_argument("-s", type=int, dest="strip_charge_cut", default=1, help="want charge cut")
parser.add_argument("-r", dest="raw_file", default='/scratch/nandan/inputfile_for_prehlt/HIEphemeralHLTPhysics_RAW/flatntuple_step3_RAW2DIGI_L1Reco_RECO_raw_wchargecut.root', help="file for raw data")

options = parser.parse_args()
barycenter_bit = options.barycenter_bit
number = options.number
threads = options.threads
width_bit = options.width_bit
avgCharge_bit = options.avgCharge_bit

def replace_line(infile, replaces_to_vals):

    with open(infile, 'r') as f:
      lines = f.readlines()

    with open(infile, 'w') as f:
        for line in lines:
            for replace_to_val in replaces_to_vals:
                replace = replace_to_val[0]
                val = replace_to_val[1]
                if replace in line:
                   line = line.replace(replace, val)
                   break 
            f.write(line)

run_cmd = 'scram b -j 8'
print(run_cmd)
os.system(run_cmd)

### hlt ###

os.system('cp step2_L1REPACK_HLT_rawp.py step2_L1REPACK_HLT_rawp_copy.py')
if not options.strip_charge_cut:
  replace_line('step2_L1REPACK_HLT_rawp_copy.py',
              [  ("process.hltSiStripClusterizerForRawPrime.Clusterizer.clusterChargeCut.refToPSet_='HLTSiStripClusterChargeCutTight'", "process.hltSiStripClusterizerForRawPrime.Clusterizer.clusterChargeCut.refToPSet_='HLTSiStripClusterChargeCutNone'"),
                  ("process.ClusterShapeHitFilterESProducer.clusterChargeCut.refToPSet_='HLTSiStripClusterChargeCutTight'", "process.ClusterShapeHitFilterESProducer.clusterChargeCut.refToPSet_='HLTSiStripClusterChargeCutNone'")
              ]) 
replace_line('step2_L1REPACK_HLT_rawp_copy.py',
              [
                 ('input = cms.untracked.int32(500)', f'input = cms.untracked.int32({number})')
              ]) 
run_cmd = 'cmsRun step2_L1REPACK_HLT_rawp_copy.py &> step2_L1REPACK_HLT_rawp.log'
print(run_cmd)
os.system(run_cmd)
output_prehlt = f'outputPhysicsHIPhysicsRawPrime0_barycenter_{barycenter_bit}_width_{width_bit}.root'
os.system(f'mv step2_L1REPACK_HLT.root {output_prehlt}')
#os.system('rm step2_L1REPACK_HLT_rawp_copy.py')

#### object comparison ####

run_cmd = f"edmEventSize -v {output_prehlt} > size.log"
print(run_cmd)
os.system(run_cmd)

### reco step ####

os.system('cp step3_RAW2DIGI_L1Reco_RECO_rawp.py step3_RAW2DIGI_L1Reco_RECO_rawp_copy.py')
replace_line('step3_RAW2DIGI_L1Reco_RECO_rawp_copy.py',
              [  
                 ('step2_L1REPACK_HLT.root', output_prehlt)
                 #('input = cms.untracked.int32(500)', f'input = cms.untracked.int32({number})')
              ])
run_cmd = 'cmsRun step3_RAW2DIGI_L1Reco_RECO_rawp_copy.py &> step3_RAW2DIGI_L1Reco_RECO_rawp.log'
print(run_cmd)
os.system(run_cmd)

output_step_reco = f'step_reco_RAW2DIGI_L1Reco_RECO_barycenter_{barycenter_bit}_width_{width_bit}.root'
os.system(f'mv step3_RAW2DIGI_L1Reco_RECO_rawp.root {output_step_reco}')
#os.system('rm step3_RAW2DIGI_L1Reco_RECO_rawp_copy.py')

#### flat ntuple ####

raw_file = options.raw_file if options.strip_charge_cut else '/scratch/nandan/inputfile_for_prehlt/HIEphemeralHLTPhysics_RAW/flatntuple_step3_RAW2DIGI_L1Reco_RECO_raw_wochargecut.root' 
run_cmd = f'python3 run_flatNtuplizer.py -rp {output_step_reco} -r {raw_file} -c -n {number}' if options.cluster\
         else f'python3 run_flatNtuplizer.py -rp {output_step_reco} -r {raw_file} -n {number}'
print(run_cmd)
os.system(run_cmd)
