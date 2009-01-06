#! /usr/bin/env python

comment= '''
The cards from 0 to 250 GeV for all the H->ZZ->4l channels, separated. To be 
combined in a card that implements include statements.
'''

# Format: mass, [sig yield, bkg yield]

HZZ_4mu_data={'115':['2.13','0.92'],
              '120':['4','1.15'],
              '130':['12.45','2.06'],
              '140':['23.22','2.65'],
              '150':['28.09','2.42'],
              '160':['14.25','3.01'],
              '170':['6.32','3.63'],
              '180':['14.54','7.1'],
              '190':['54.95','17'],
              '200':['62.78','19.93'],
              '250':['54.48','21.62']}

HZZ_2mu2e_data={'115':['2.7','1.4'],
                '120':['5.72','4.37'],
                '130':['13.75','1.69'],
                '140':['35.16','6'],
                '150':['42.32','6.34'],
                '160':['23.52','6.11'],
                '170':['11.42','8.60'],
                '180':['25.97','12.06'],
                '190':['109.6','48.12'],
                '200':['109.3','48.60'],
                '250':['87.2','40.69']}

HZZ_4e_data={'115':['1.52','2.26'],
             '120':['2.97','1.94'],
             '130':['8.18','3.71'],
             '140':['15.80','4.31'],
             '150':['17.19','3.68'],
             '160':['8.38','3.10'],
             '170':['3.76','3.37'],
             '180':['9.95','6.42'],
             '190':['34.12','14.69'],
             '200':['38.20','17.29'],
             '250':['27.68','13.40']}

# systematics in percent

HZZ_bkg_sys={'115':'0.031',
             '120':'0.036',
             '130':'0.033',
             '140':'0.035',
             '150':'0.037',
             '160':'0.038',
             '170':'0.04',
             '180':'0.04',
             '190':'0.04',
             '200':'0.041',
             '250':'0.052'}

# data dict
channel_dict = {'4mu':HZZ_4mu_data,
                '2mu2e':HZZ_2mu2e_data,
                '4e':HZZ_4e_data}


# Loop masses and inside, on the channels:

for mass in HZZ_bkg_sys.keys(): # systematics taken, masses are the same!
    for channel,data_dict in channel_dict.items():
        sig_yield,bkg_yield=data_dict[mass]
        sys=HZZ_bkg_sys[mass]
        card= '''
################################################################################
# H -> ZZ -> '''+channel+'''
################################################################################

[hzz_'''+channel+''']
    variables = x
    x = 0 L(0 - 1)

[hzz_'''+channel+'''_sig]
    yield_factors_number = 2

    yield_factor_1 = sig_scale
    sig_scale = 1 C

    yield_factor_2 = sig_hzz_'''+channel+'''
    sig_hzz_'''+channel+''' = '''+sig_yield+''' C

[hzz_'''+channel+'''_sig_x]
    model = yieldonly

[hzz_'''+channel+'''_bkg]
    yield_factors_number = 2

    yield_factor_1 = bkg_scale
    bkg_scale = 1 L (0 - 100)
    bkg_scale_constraint = Gaussian,1,'''+sys+'''

    yield_factor_2 = bkg_hzz_'''+channel+'''
    bkg_hzz_'''+channel+''' = '''+bkg_yield+''' C


[hzz_'''+channel+'''_bkg_x]
    model = yieldonly

'''

# 3 channels

        ofile=open('HZZ_%s_%s_card.rsc' %(channel,mass),'w')
        ofile.write(card)
        ofile.close()

    combo_card='''

################################################################################
# H -> ZZ -> 4l combination for mass '''+mass+''' GeV
################################################################################

// Includes
include HZZ_4mu_'''+mass+'''_card.rsc
include HZZ_4e_'''+mass+'''_card.rsc
include HZZ_2mu2e_'''+mass+'''_card.rsc

// Here we specify the names of the models built down in the card that we want
// to be combined

[hzz4l]
    model = combined
    components = hzz_4mu, hzz_4e, hzz_2mu2e

[hzz4l_2channels]
    model = combined
    components = hzz_4mu, hzz_2mu2e

[hzz4mu_only]
    model = combined
    components = hzz_4mu

[hzz2mu2e_only]
    model = combined
    components = hzz_2mu2e

[hzz4e_only]
    model = combined
    components = hzz_4e

'''
    combo_ofile=open('HZZ_combination_%s.rsc' %mass,'w')
    combo_ofile.write(combo_card)
    combo_ofile.close()

