# qqH , H -> tau tau analysis (PTDR, 30fb-1)
# Pag 293, PTDR Physics Performance Volume II
# Hmass = 135 GeV
#
# The Card is modified to obtain exemple -2lnQ distributions:
# for that the signal yield is brought from 10.33 to 3.
#
# Signal
#  - Starting xsec 82.38
#  - Nevts = 10.3
# Bkg
#  1. Z(gamma*) from EW+QCD Z+2/3jets background
#    - Starting xsec 1914
#    - Nevts 23.5
#  2. ttbar->WbWb + W+3/4jets background
#    - Starting xsec 100.45 * 10**3
#    - Nevts 8.3
#
# To build a nice example for RSC the yiled of the signal was brought down to 
# 3 events. It is clearly NOT what is stated in the TDR. 
# This is a didactical example!!
#
#
# The Datacard:
#
# The datacard is divided in sections. Each section begins with a 
# [saection_name], which controlos the flow.
# The names of the sections contain crucial information for the creation of the
# model. This information can be the name of the variable, the "sig" or "bkg" 
# identifier, the component index.
#
# Each single model presents the following features:
#  - A variable or multiple variables.
#  - A top level signal and background.
#  - A description of the signal(s)/background(s) components.
#
# Multiple single models can be combined in a Combined model
#
# As you can see in the following card, the background or the signal can have
# multiple components. You just have to declare their number and then treat 
# them one by one.
#
# ALL the parameters can be affected by a systematic uncertainity. To describe 
# it, just add a new line with the same name of the parameter and the 
# "_constraint" suffix. For example:
#
# parameter_name = 123 L (100 - 200)
# parameter_name_constraint = Gaussian,123,0.50
#
# To know more about the constraints, see the Constraint and RscCombinedModel 
# classes documentation.
#
# Yields in the signal/background top sections are forseen to be expressed also 
# as a product of multiple terms. 
# For a single term:
#    qqhtt_single_sig_yield = 3 C
#
# For a product:
#     yield_factors_number = 2
#
#    yield_factor_1 = scale
#    scale = 1 L (0 - 20)
#    scale_constraint = LogNormal,1,0.078
#
#    yield_factor_2 = bkg1
#    bkg1 = 23.5 C
#


# The Combined Model------------------------------------------------------------

[qqhtt]
    model = combined
    components = qqhtt_single


# The single model--------------------------------------------------------------

[qqhtt_single]
    variables = mh
    mh = 100 L(30 - 250)  // [GeV/c^{2}] 
# The L (30 -250) is the interval in which the variable is allowed to vary. Useful for teh fits!
# If a C is present, this means that the parameter is constant.
# The last "// [GeV/c^{2}]" is the unit to assign to the quantity
#

# ----- top level signal / background -----

[qqhtt_single_sig]
    number_components = 1
    qqhtt_single_sig_yield = 3 C
# The original Value is 10.33 C

[qqhtt_single_bkg]
number_components = 2

[qqhtt_single_bkg1]
    yield_factors_number = 2

    yield_factor_1 = scale
    scale = 1 L (0 - 20)
    scale_constraint = LogNormal,1,0.078

    yield_factor_2 = bkg1
    bkg1 = 23.5 C

[qqhtt_single_bkg2]
    yield_factors_number = 2

    yield_factor_1 = scale
    scale = 1 L (0 - 20)
    scale_constraint = LogNormal,1,0.078

    yield_factor_2 = bkg2
    bkg2 = 8.2 C


# ----- signal distribution(s) -----

[qqhtt_single_sig_mh]
    model = gauss
    qqhtt_single_sig_mh_mean  = 140.702 C
    qqhtt_single_sig_mh_sigma = 12.8216 C

# ----- background distribution(s) -----

[qqhtt_single_bkg1_mh]
    model = BreitWigner
    qqhtt_single_bkg1_mh_mean  = 97.165 C
    qqhtt_single_bkg1_mh_width = 17.1847 C

[qqhtt_single_bkg2_mh]
    model = poly7
    qqhtt_single_bkg2_mh_coef1 = -6.1666e+10 C
    qqhtt_single_bkg2_mh_coef2 =  3.0537e+09 C
    qqhtt_single_bkg2_mh_coef3 = -2.1112e+07 C
    qqhtt_single_bkg2_mh_coef4 =  6.3835e+04 C
    qqhtt_single_bkg2_mh_coef5 = -9.8900e+01 C
    qqhtt_single_bkg2_mh_coef6 =  7.4581e-02 C
    qqhtt_single_bkg2_mh_coef7 = -1.9461e-05 C
