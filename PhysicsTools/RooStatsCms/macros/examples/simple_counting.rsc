[my_combination]
    model = combined
    components = my_analysis

#section 1
[my_analysis]
    variables = x
    x = 0 L(0 - 1)

#section 2
[my_analysis_sig]
    my_analysis_sig_yield = 100 L (0 - 1000)

#section 3
[my_analysis_sig_x]
    model = yieldonly

#section 4
[my_analysis_bkg]
    my_analysis_bkg_yield = 1000 C

#section 5
[my_analysis_bkg_x]
    model = yieldonly
