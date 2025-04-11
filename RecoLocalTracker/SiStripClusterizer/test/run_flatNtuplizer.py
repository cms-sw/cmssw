import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-r", dest="input_raw", default='/home/users/nandan/backup/flatntuple_step5_RAW2DIGI_L1Reco_RECO_wchargecut.root', help="flatntuple for raw")
parser.add_argument("-rp", dest="input_rawp", default='', help="input for rawp")
parser.add_argument("-n", dest="n", default='-1', help="number of events to be run")
parser.add_argument("-c", action='store_true', dest="cluster", default=False, help="make flatntuple for cluster")

options = parser.parse_args()

input_for_raw = options.input_raw

input_for_rawp = options.input_rawp
output_for_rawp = f'flatntuple_{input_for_rawp}'
cmd = f'cmsRun sep19_2_1_dump_rawprime.py inputFiles={input_for_rawp} outputFile={output_for_rawp} n={options.n} c={options.cluster} &> flatNtuple.log'
print(cmd)
os.system(cmd)

### cluster ###
if options.cluster:
  os.system('g++ `root-config --cflags --glibs` -O3 rootMacro/LHCC_rawprime_clusters.cc -o rootMacro/LHCC_rawprime_clusters.o')
  os.system(f'./rootMacro/LHCC_rawprime_clusters.o flatntuple_{input_for_rawp} {input_for_raw} &> cluster.log')

### objects ###
os.system('g++ `root-config --cflags --glibs` -O3 rootMacro/LHCC_raw_vs_rawprime.cc -o rootMacro/LHCC_raw_vs_rawprime.o')
os.system(f'./rootMacro/LHCC_raw_vs_rawprime.o flatntuple_{input_for_rawp} {input_for_raw} &> object.log')
