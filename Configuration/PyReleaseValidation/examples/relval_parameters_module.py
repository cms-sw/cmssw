
#############################################################
#                                                           #
#              relval_parameters_module                     #
#                                                           #
#  This module contains a dictionary in which               #
#  the parameters relevant for the process are stored.      #
#  The parameters are:                                      #
#   - Type of the events          (string)                  #
#   - Number of the events        (int)                     # 
#   - Energy of the events        (string)                  #
#   - input and output files      (string)                  #
#   - Step: SIM DIGI RECO ALL     (string)                  #
#  The supported types are:                                 #
#   - QCD (energy in the form min_max)                      #
#   - B_JETS, C_JETS (energy in the form min_max for cuts)  #
#   - TTBAR                                                 #
#   - MU+,MU-,E+,E-,GAMMA,10MU+,10E-...                     #
#   - TAU (energy in the form min_max for cuts)             #
#   - HZZEEEE, HZZMUMUMUMU                                  #
#   - ZEE (no energy is required)                           #
#   - ZPJJ: zee prime in 2 jets                             #
#                                                           #
#############################################################

# Process Parameters

# The name of the process
process_name='ALL'
# The type of the process. Please see the complete list of 
# available processes.
evt_type='MU+'
# The energy in GeV. Some of the tipes require an
# energy in the form "Emin_Emax"
energy='10'
# Number of evts to generate
evtnumber=10
# Input and output file names
infile_name=''
outfile_name='MU+_10_ALL.root'
# The step: SIM DIGI RECO and ALL to do the 3 in one go.
step='ALL'
# Omit the output in a root file
output_flag= True

# Pyrelval parameters
# Enable verbosity
dbg_flag=True
# Dump the oldstyle cfg file.
dump_cfg_flag=False
