#!/usr/bin/env python

""" This script can be used to run manually the dcs o2o over an interval of time,
dividing it in smaller intervals. Running on an interval too big can cause
a result of the query to the database so big that the machine runs out of memory.
By splitting it in smaller intervals of a given DeltaT it is possible to keep
under control the memory used.
"""

import os
import datetime
import subprocess
import argparse

def insert_to_file(template, target, replace_dict):
    '''Update the template file based on the replace_dict, and write to the target.'''
    with open(template, 'r') as input_file:
        config=input_file.read()
    with open(target, 'w') as output_file:
        for key, value in replace_dict.iteritems():
            config = config.replace(key, value)
        output_file.write(config)


def main():
    parser = argparse.ArgumentParser(description='Run SiStrip DCS O2O by splitting into small intervals.')
    parser.add_argument('-i', '--interval',
        dest = 'interval',
        type = int,
        default = 1,
        help = 'Interval (in hours) for splitting jobs. Default: %(default)d hours',
    )
    parser.add_argument('-b', '--begin',
        dest = 'begin',
        default = '2016-01-01 00:00:00',
        help = 'Beginning time of the interval. Format: [YYYY-MM-DD HH:MM:SS]. Default: %(default)s',
    )
    parser.add_argument('-e', '--end',
        dest = 'end',
        default = '2016-02-01 00:00:00',
        help = 'End time of the interval. Format: [YYYY-MM-DD HH:MM:SS]. Default: %(default)s',
    )
    parser.add_argument('-t', '--template',
        dest = 'template',
        default = 'dcs_o2o_template_cfg.py',
        help = 'Template config file. Default: %(default)s',
    )    
    parser.add_argument('--db',
        dest = 'dbfile',
        default = 'SiStripDetVOff.db',
        help = 'Output tag. Default: %(default)s',
    )    
    parser.add_argument('--tag',
        dest = 'tag',
        default = 'SiStripDetVOff',
        help = 'Output tag. Default: %(default)s',
    )    
    args = parser.parse_args()

    # initialize the sqlite file
    if not os.path.exists(args.dbfile):
        with open(args.dbfile, 'w'):
            pass

    dt_begin = datetime.datetime.strptime(args.begin, '%Y-%m-%d %H:%M:%S')
    dt_end   = datetime.datetime.strptime(args.end,   '%Y-%m-%d %H:%M:%S')
    
    while (dt_end-dt_begin).total_seconds() > 0:
        dt_next = dt_begin + datetime.timedelta(hours=args.interval)
        tmin_str = dt_begin.strftime('%Y, %-m, %-d, %-H, 0, 0, 0')
        tmax_str = dt_next.strftime('%Y, %-m, %-d, %-H, 0, 0, 0')
        targetFile = 'dcs_%s_to_%s_cfg.py' % (dt_begin.strftime('%Y-%m-%d__%H_%M_%S'), dt_next.strftime('%Y-%m-%d__%H_%M_%S'))
        replace_dict={'_TMIN_':tmin_str, '_TMAX_':tmax_str, '_DBFILE_':args.dbfile, '_TAG_':args.tag}
        insert_to_file(args.template, targetFile, replace_dict)
        print 'Running %s' % targetFile
        command = 'cmsRun %s' % targetFile
        #print command
        subprocess.Popen(command, shell=True).communicate()
        print '='*50
        dt_begin = dt_next

if __name__ == '__main__':
    main()
