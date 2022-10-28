from __future__ import print_function
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
              'cms_trg_r/@cms_omds_adg'
             ]

if hash( sqlplusCmd[-1] ) != 1687624727082866629:
    print('Do not forget to plug password to this script')
    print('Exiting.')
    exit(0)

myre = re.compile(r'(ID)|(-{80})')

# if no arguments are given, query the top level keys only and exit
if len(sys.argv) == 1:
    sqlplus = subprocess.Popen(sqlplusCmd, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    print('No args specified, querying and printing only top-level config keys:')
    for line in re.split('\n',sqlplus.communicate('select unique ID from CMS_TRG_L1_CONF.BMTF_KEYS;')[0]):
        if myre.search(line) == None :
            print(line)
    sqlplus = subprocess.Popen(sqlplusCmd, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    print('No args specified, querying and printing only top-level Run Settings keys:')
    for line in re.split('\n',sqlplus.communicate('select unique ID from CMS_TRG_L1_CONF.BMTF_RS_KEYS;')[0]):
        if myre.search(line) == None :
            print(line)
    print('Pick any of these keys as an argument next time you run this script')
    exit(0)

# if an argument is given query the whole content of the key
key = sys.argv[1]
rsKey = sys.argv[2]

# if the argument is the very top level key, querry the BMTF specific subkey
queryKey   = "select BMTF_KEY    from CMS_TRG_L1_CONF.L1_TRG_CONF_KEYS where ID='{0}'".format(key)
queryRsKey = "select BMTF_RS_KEY from CMS_TRG_L1_CONF.L1_TRG_RS_KEYS   where ID='{0}'".format(rsKey)

sqlplus = subprocess.Popen(sqlplusCmd, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
for line in re.split('\n',sqlplus.communicate(queryKey+';')[0]):
    print(line)
    if re.search('/v',line) :
        key=line

sqlplus = subprocess.Popen(sqlplusCmd, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
for line in re.split('\n',sqlplus.communicate(queryRsKey+';')[0]):
    print(line)
    if re.search('/v',line) :
        rsKey=line

print(key+":"+rsKey)

queryKeys = """
            select
                CONF.HW as HW, CONF.ALGO as ALGO, RS.MP7 as MP7, RS.AMC13 as AMC13
            from
                CMS_TRG_L1_CONF.BMTF_KEYS CONF, CMS_TRG_L1_CONF.BMTF_RS_KEYS RS
            where
                CONF.ID = '{0}' and RS.ID = '{1}'
            """.format(key,rsKey)

# write results for specific configs to the following files
batch = {
         'HW'     : 'hw.xml',
         'ALGO'   : 'algo.xml',
         'MP7'    : 'mp7.xml',
         'AMC13'  : 'amc13.xml'
        }

# do the main job here
for config,fileName in batch.items():

    sqlplus = subprocess.Popen(sqlplusCmd, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    with open(fileName,'w') as f:
        query = """
                select
                    CLOBS.CONF
                from
                    CMS_TRG_L1_CONF.BMTF_CLOBS CLOBS, ({0}) KEY
                where
                    CLOBS.ID = KEY.{1}
                """.format(queryKeys, config)

        for line in re.split('\n',sqlplus.communicate('\n'.join(['set linesize 200', 'set longchunksize 200000 long 200000 pages 0',query+';']))[0]):
            f.write('\n')
            f.write(line)
        f.close()


sqlplus = subprocess.Popen(sqlplusCmd, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
print('Following keys were found:')
for line in re.split('\n',sqlplus.communicate(queryKeys+';')[0]):
    print(line)

print('Results are saved in ' + ' '.join(batch.values()) + ' files')

