#!/usr/bin/env python

import subprocess

iovs = [
    'JetCorrectorParametersCollection_Jec11_V10_AK4Calo',
    'JetCorrectorParametersCollection_Jec11_V10_AK4PF',
    'JetCorrectorParametersCollection_Jec11_V10_AK4PFchs',
    'JetCorrectorParametersCollection_Jec11_V10_AK4JPT',
    'JetCorrectorParametersCollection_Jec11_V5_AK4TRK',
    'JetCorrectorParametersCollection_Jec11_V10_AK8Calo',
    'JetCorrectorParametersCollection_Jec11_V10_AK8PF',
    'JetCorrectorParametersCollection_Jec11_V10_AK8JPT',
    'JetCorrectorParametersCollection_Jec11_V10_KT4Calo',
    'JetCorrectorParametersCollection_Jec11_V10_KT4PF',
    'JetCorrectorParametersCollection_Jec11_V5_KT6Calo',
    'JetCorrectorParametersCollection_Jec11_V5_KT6PF',
    'JetCorrectorParametersCollection_Jec11_V5_AK4Calo',
    'JetCorrectorParametersCollection_Jec11_V5_AK4PF'
    
    
    ]

for iov in iovs :
    s = 'cmscond_list_iov -c sqlite_file:Jec11_V10.db -t ' + iov
    subprocess.call( [s, ""], shell=True )    
