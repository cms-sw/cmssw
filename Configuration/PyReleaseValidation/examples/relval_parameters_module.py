
#############################################################
#                                                           #
#             + relval_parameters_module +                  #
#                                                           #
#  The supported types are:                                 #
#                                                           #
#   - QCD (energy in the form min_max)                      #
#   - B_JETS, C_JETS, UDS_JETS (energy in the form min_max) #
#   - TTBAR                                                 #
#   - BSJPSIPHI                                             #
#   - MU+,MU-,E+,E-,GAMMA,10MU+,10E-...                     #
#   - TAU (energy in the form min_max for cuts)             #
#   - HZZEEEE, HZZMUMUMUMU                                  #
#   - ZEE (no energy is required)                           #
#   - ZPJJ: zee prime in 2 jets                             #
#                                                           #
#############################################################

# Process Parameters

# The name of the process
process_name='SIM'
# The type of the process. Please see the complete list of 
# available processes.
evt_type='QCD'
# The energy in GeV. Some of the tipes require an
# energy in the form "Emin_Emax"
energy='380_470'
# Number of evts to generate
evtnumber=1
# Input and output file names
infile_name=''
outfile_name='QCD_380_470_SIM.root'
# The step: SIM DIGI RECO and ALL to do the 3 in one go.
step='SIM'
# Omit the output in a root file
output_flag=True
# Use the profiler service
profiler_service_cuts=''

# Pyrelval parameters
# Enable verbosity
dbg_flag=True
# Dump the oldstyle cfg file.
dump_cfg_flag=False
