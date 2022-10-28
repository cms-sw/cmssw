from __future__ import print_function
# select CALOL1_KEY from CMS_TRG_L1_CONF.L1_TRG_CONF_KEYS where ID='collisions2016_TSC/v206' ;
import re
import os, sys, shutil
import subprocess
"""
A simple helper script that provided with no arguments dumps a list of
top-level keys, and provided with any key from this list as an argument,
dumps a list of sub-keys and saves corresponding configuration to local
files.
"""

# connection string
sqlplusCmd = ['env',
              'sqlplus',
              '-S',
              'cms_trg_r/X3lmdvu4@cms_omds_adg'
             ]

if hash( sqlplusCmd[-1] ) != 1687624727082866629:
    print('Do not forget to plug password to this script')
    print('Exiting.')
    exit(0)

myre = re.compile(r'(ID)|(-{80})')

# if no arguments are given, query the top level keys only and exit
if len(sys.argv) == 1:
    sqlplus = subprocess.Popen(sqlplusCmd, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    print('No args specified, querying and printing only top-level keys:')
    for line in re.split('\n',sqlplus.communicate('select unique ID from CMS_TRG_L1_CONF.CALOL2_KEYS;')[0]):
        if myre.search(line) == None :
            print(line)
    print('Pick any of these keys as an argument next time you run this script')
    exit(0)

# if an argument is given query the whole content of the key
key = sys.argv[1]

sqlplus = subprocess.Popen(sqlplusCmd,
                           shell=False,
                           stdout=subprocess.PIPE,
                           stdin=subprocess.PIPE
                          )

queryKey = "select CALOL1_KEY from CMS_TRG_L1_CONF.L1_TRG_CONF_KEYS where ID='{0}'".format(key)

for line in re.split('\n',sqlplus.communicate(queryKey+';')[0]):
    print(line)
    if re.search('/v',line) :
        key=line

print(key)

queryKeys = """
            select
                HW, ALGO, INFRA
            from
                CMS_TRG_L1_CONF.CALOL1_KEYS
            where
                ID = '{0}'
            """.format(key)

# write results for specific configs to the following files
batch = {
         'HW'    : 'hw.xml',
         'ALGO'  : 'algo.xml',
         'INFRA' : 'infra.xml'
        }

# do the main job here
for config,fileName in batch.items():
    sqlplus = subprocess.Popen(sqlplusCmd, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

    with open(fileName,'w') as f:
        query = """
                select
                    CONF.CONF
                from
                    CMS_TRG_L1_CONF.CALOL1_CLOBS CONF, ({0}) KEY
                where
                    CONF.ID = KEY.{1}
                """.format(queryKeys, config)

        for line in re.split('\n',sqlplus.communicate('\n'.join(['set linesize 200', 'set longchunksize 2000000 long 2000000 pages 0',query+';']))[0]):
            f.write('\n')
            f.write(line)
        f.close()

sqlplus = subprocess.Popen(sqlplusCmd, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
print('Following keys were found:')
for line in re.split('\n',sqlplus.communicate(queryKeys+';')[0]):
    print(line)


print('Results are saved in ' + ' '.join(batch.values()) + ' files')


