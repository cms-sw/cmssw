#!/usr/bin/env python

from optparse import OptionParser, OptionGroup

## set up the option parser
parser = OptionParser(usage="usage: %prog [options] ARG",
                      description="This is a script to comment uncertainties, which are element of a list of uncertainties to be dropped in an actual set of datacards.")
parser.add_option("--drop-list", dest="drop_list", default="", type="string",
                  help="Path to a list of uncertainties to be dropped/commented from the datacards. [Default: \"\"]")
(options, args) = parser.parse_args()
## check number of arguments; in case print usage
if len(args) < 1 :
    parser.print_usage()
    exit(1)

import os

drop_uncerts = []
file = open(options.drop_list, 'r')
for line in file :
    drop_uncerts.append(line[:line.rfind('\n')])
file.close()

for card in os.listdir(args[0]) :
    if not card.endswith('.txt') :
        continue
    path = args[0]+'/'+card
    old_file = open(path, 'r')
    new_file = open(path+'_tmp', 'w')
    for line in old_file :
        words = line.lstrip().split()
        if words[0] in drop_uncerts :
            line = '#'+line
        new_file.write(line)
    os.system('mv {NEW} {OLD}'.format(NEW=path+'_tmp', OLD=path))
