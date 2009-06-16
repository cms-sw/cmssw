[my_combination]
    model = combined
    components = my_analysis

#section 1
[my_analysis]
    variables = x
    x = 0 L(0 - 300)

#section 2
[my_analysis_sig]
    my_analysis_sig_yield = 40 L (0 - 1000)

#section 3
[my_analysis_sig_x]
    model = gauss
    my_analysis_sig_x_mean = 70 C
    my_analysis_sig_x_sigma = 10 C

#section 4
[my_analysis_bkg]
    my_analysis_bkg_yield = 500 C

#section 5
[my_analysis_bkg_x]
    model = exponential
    my_analysis_bkg_x_slope = -0.01 C
