import os
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("-n", dest="number", default='100', help="how many numbers of events")
parser.add_argument("-b", dest="barycenter_bits", nargs='+', default=[16, 10, 12, 14, 8, 6, 4], help="bit to be studied")
parser.add_argument("-w", dest="width_bits", nargs='+', default=[], help="bit to be studied for width")
parser.add_argument("-a", dest="avgCharge_bits", nargs='+', default=[], help="bit to be studied for avgCharge")
parser.add_argument("-t", dest="threads", default=20, help="# of threads")
parser.add_argument("-r", type=int, dest="remove", default=0, help="want to delete directory")
parser.add_argument("-C", action='store_false', dest="cms_command", default=1, help="either want to run cmsDriver commands or only the .cc file")
parser.add_argument("-c", type=int, dest="cluster", default=1, help="want flatntuple for cluster")
parser.add_argument("-o", dest="output", default='output', help="output directory name")
parser.add_argument("-p", type=int, dest="parallel", default=5, help="how many runs you want at the same time ")
parser.add_argument("-g", dest="git_branch", default='saswati', help="which git branch you want")
parser.add_argument("-s", type=int, dest="strip_charge_cut", default=1, help="want to add chage cut")

options = parser.parse_args()
barycenter_bits = options.barycenter_bits
width_bits = options.width_bits
avgCharge_bits = options.avgCharge_bits
output = options.output

with open('makefile', 'w') as f:

    allbit = ' '.join([str(b)+'bit'+'_'+str(w)+'bit'+'_'+str(a)+'bit' for b in barycenter_bits for w in width_bits for a in avgCharge_bits])
    allbit_plot = allbit + ' plot'
    f.write(f'all: {allbit_plot}\n')
    for bb in barycenter_bits:
      for wb in width_bits:
         for ab in avgCharge_bits:
           f.write(f'{bb}bit_{wb}bit_{ab}bit:\n')
           output = f'{options.output}_barycenter_{bb}bit_width_{wb}bit_avgCharge_{ab}bit'
           if options.cms_command:
              f.write(f'\tbash build_env.sh {bb}bit {wb}bit {ab}bit {options.number} {options.threads} {options.remove} {output} {options.cluster} {options.git_branch} {options.strip_charge_cut}\n')
           else:
            outputdir = os.path.join(f'/scratch/{os.getlogin()}', output)
            pwd       = os.getcwd()
            if options.cluster:
              f.write(f'\tcd {outputdir} && {pwd}/rootMacro/LHCC_rawprime_clusters.o flatntuple_step_reco_RAW2DIGI_L1Reco_RECO_barycenter_{bb}bit_width_{wb}bit.root ~/backup/flatntuple_step5_RAW2DIGI_L1Reco_RECO_wchargecut.root > cluster.log && {pwd}/rootMacro/LHCC_raw_vs_rawprime.o flatntuple_step_reco_RAW2DIGI_L1Reco_RECO_barycenter_{bb}bit_width_{wb}bit.root ~/backup/flatntuple_step5_RAW2DIGI_L1Reco_RECO_wchargecut.root > object.log\n')
            else:
                f.write(f'\tcd {outputdir} && {pwd}/rootMacro/LHCC_raw_vs_rawprime.o flatntuple_step_reco_RAW2DIGI_L1Reco_RECO_barycenter_{bb}bit_width_{wb}bit.root ~/backup/flatntuple_step5_RAW2DIGI_L1Reco_RECO_wchargecut.root > object.log\n')
    f.write(f'\nplot: {allbit}\n')
    barycenter_bits = ' '.join([b for b in barycenter_bits])
    width_bits      = ' '.join([w for w in width_bits])
    avgCharge_bits  = ' '.join([a for a in avgCharge_bits])
    f.write(f'\t python3 plot_comparison.py -b {barycenter_bits} -w {width_bits} -a {avgCharge_bits} -o {options.output}')

os.system(f'make -f makefile -j {options.parallel}')
 
