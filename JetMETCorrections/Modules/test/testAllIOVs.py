#!/usr/bin/env python

import subprocess

iovs = [
    'JetCorrectorParametersCollection_Jec11_V10_AK5Calo',
    'JetCorrectorParametersCollection_Jec11_V10_AK5PF',
    'JetCorrectorParametersCollection_Jec11_V10_AK5PFchs',
    'JetCorrectorParametersCollection_Jec11_V10_AK5JPT',
    'JetCorrectorParametersCollection_Jec11_V5_AK5TRK',
    'JetCorrectorParametersCollection_Jec11_V10_AK7Calo',
    'JetCorrectorParametersCollection_Jec11_V10_AK7PF',
    'JetCorrectorParametersCollection_Jec11_V10_AK7JPT',
    'JetCorrectorParametersCollection_Jec11_V10_KT4Calo',
    'JetCorrectorParametersCollection_Jec11_V10_KT4PF',
    'JetCorrectorParametersCollection_Jec11_V5_KT6Calo',
    'JetCorrectorParametersCollection_Jec11_V5_KT6PF',
    'JetCorrectorParametersCollection_Jec11_V5_IC5Calo',
    'JetCorrectorParametersCollection_Jec11_V5_IC5PF'
    
    
    ]

for iov in iovs :
    s = 'cmscond_list_iov -c sqlite_file:Jec11_V10.db -t ' + iov
    subprocess.call( [s, ""], shell=True )    
