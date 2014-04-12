#!/usr/bin/env python

import os
import castor
import random

random.seed(1)

#LOCAL_DIR = '/tmp/friis/MVATraining/signal'
LOCAL_DIR = '/data2/friis/MVATraining/signal'
if not os.path.exists(LOCAL_DIR):
    os.makedirs(LOCAL_DIR)

castor_signal_directories = [os.path.join(
    '/castor/cern.ch/user/f/friis',
    'TaNCTrainingV3', x) for x in [
        'crab_sig_A120',
        'crab_sig_A140',
        'crab_sig_A160',
        'crab_sig_A180',
        'crab_sig_A200',
        'crab_sig_A250',
        'crab_sig_A300',
        'crab_sig_Ztautau_powhegZ2',
    ]]

castor_file_list = []
local_file_list = []

for castor_dir in castor_signal_directories:
    for file in castor.nslsl(castor_dir + '/'):
        local_file = os.path.join(LOCAL_DIR, file['file'])
        if not os.path.exists(local_file):
            castor_file_list.append(file['path'])
        else:
            local_file_list.append(local_file)

with open('copy_castor.txt', 'w') as castor_copy_file:
    for castor_file in castor_file_list:
        castor_copy_file.write('%s %s\n' % (castor_file, LOCAL_DIR))

def is_test_file(fraction=0.2):
    if random.random() < fraction:
        return True
    else:
        return False

random.shuffle(local_file_list)
with open('signalfiles.list', 'w') as trainfiles:
    with open('signalfiles.test', 'w') as testfiles:
        for file in local_file_list:
            if 'training' not in file:
                continue
            write_to = is_test_file(0.2) and testfiles or trainfiles
            write_to.write("file:%s\n" % file)


