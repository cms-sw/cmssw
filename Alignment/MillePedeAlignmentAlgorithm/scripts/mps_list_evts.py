#!/usr/bin/env python

""" Print the total number of events processed by the mille jobs per dataset

The information is taken from the `mps.db' file. Will group entries of the
same dataset and also datasets the script *thinks* belong to the same
data type, e.g. 0T cosmics. This is implemented very simple and should 
always be checked by the user.

Usage:

 `python mps_list_evts.py <mps.db file name>' or, after `scram b'
 `mps_list_evts.py <mps.db file name>'

M. Schroeder, DESY Hamburg      26-May-2014
"""

import sys


mps_db = "mps.db"               # the mps.db file, default value


def get_mille_lines():
    """ Return list of mps.db lines that correspond to a mille job """
    mille_lines = []
    with open(mps_db,"r") as db:
        for line in db:
            line = line.rstrip('\n')
            # mille and pede job lines have 13 `:' separated fields
            parts = line.split(":")
            if len(parts) == 13:
                # mille lines start with `<123>:job<123>'
                if parts[1] == "job"+parts[0]:
                    mille_lines.append(parts)

    return mille_lines



def get_num_evts_per_dataset(mille_lines):
    """ Return number of events per dataset

    Returns a dict `<dataset>:<num_evts>', where <dataset> is the label
    in the last field of the mille line.
    """
    num_evts_per_dataset = {}
    for line in mille_lines:
        dataset = line[12]
        num_evts = int(line[6])
        if dataset in num_evts_per_dataset:
            num_evts_per_dataset[dataset] = num_evts_per_dataset[dataset] + num_evts
        else:
            num_evts_per_dataset[dataset] = num_evts

    return num_evts_per_dataset



def get_num_evts_per_merged_dataset(merged_datasets,num_evts_per_dataset):
    """ Return number of events per merged dataset

    Returns a dict `<merged_dataset>:<num_evts>'; see comments to function
    `merge_datasets' for an explanation of <merged_dataset>.
    """
    num_evts_per_merged_dataset = {}
    for merged_dataset,datasets in merged_datasets.iteritems():
        num_evts = 0
        for dataset in datasets:
            num_evts = num_evts + num_evts_per_dataset[dataset]
        num_evts_per_merged_dataset[merged_dataset] = num_evts

    return num_evts_per_merged_dataset



def merge_datasets(num_evts_per_dataset):
    """ Return dict `<merged_dataset> : list of <dataset>'

    Associates all datasets in `num_evts_per_dataset' that belong by their
    name to the same PD but to a different run era. For example:
    
    isolated_mu_runa_v1, isolated_mu_runb_v1, isolated_mu_runc_v2 --> isolated_mu

    The returned dict has as value a list of the merged datasets.
    """
    datasets = num_evts_per_dataset.keys()
    merged_datasets = {}
    for dataset in datasets:
        bare_name = dataset[0:dataset.find("run")].rstrip("_")
        if bare_name in merged_datasets:
            merged_datasets[bare_name].append(dataset)
        else:
            merged_datasets[bare_name] = [dataset]

    return merged_datasets



def print_merging_scheme(merged_datasets):
    """ Print number of events per merged dataset

    See comments to function `merge_datasets' for an explanation
    of what is meant by merged dataset.
    """
    print "Defining the following merged datasets:"
    for merged_dataset,datasets in merged_datasets.iteritems():
        print "\n  `"+merged_dataset+"' from:"
        for dataset in datasets:
            print "    `"+dataset+"'"



def print_num_evts_per_dataset(num_evts_per_dataset):
    """ Print number of events per dataset

    See comments to function `get_num_evts_per_dataset' for an
    explanation of what is meant by dataset.
    """
    print "The following number of events per dataset have been processed:"
    datasets = sorted(num_evts_per_dataset.keys())
    max_name = 0
    max_num = 0
    for dataset in datasets:
        if len(dataset) > max_name:
            max_name = len(dataset)
        if len(str(num_evts_per_dataset[dataset])) > max_num:
            max_num = len(str(num_evts_per_dataset[dataset]))
    expr_name = " {0: <"+str(max_name)+"}"
    expr_num = " {0: >"+str(max_num)+"}"
    for dataset in datasets:
        print expr_name.format(dataset)+" : "+expr_num.format(str(num_evts_per_dataset[dataset]))


if  __name__ == '__main__':
    """ main subroutine """

    if len(sys.argv) < 2:
        print 'ERROR'
        print 'usage:'
        print '  python mps_list_evts.py <mps.db file name>  or, after scram b'
        print '  mps_list_evts.py <mps.db file name>'
        sys.exit(1)

    mps_db = sys.argv[1]
    print 'Parsing '+mps_db

    mille_lines = get_mille_lines()
    num_evts_per_dataset = get_num_evts_per_dataset(mille_lines)
    merged_datasets = merge_datasets(num_evts_per_dataset)
    num_evts_per_merged_dataset = get_num_evts_per_merged_dataset(merged_datasets,num_evts_per_dataset)
    
    print "\n"
    print_num_evts_per_dataset(num_evts_per_dataset)
    print "\n\n"
    print_merging_scheme(merged_datasets)
    print "\n\n"
    print_num_evts_per_dataset(num_evts_per_merged_dataset)


